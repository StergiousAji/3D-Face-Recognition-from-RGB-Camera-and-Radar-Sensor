#include <cstdint>
#include <chrono>
#include <iostream>
#include <fstream>
#include <thread>
#include <csignal>
#include <string>

#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/resource.h>
#include <fcntl.h>

#include "soli.h"

#define PROFILE_DEFAULT         0
#define PROFILE_SHORT_RANGE     1
#define PROFILE_LONG_RANGE      2 

uint8_t HEADER_VERSION = 1;
uint16_t HEADER_SIZE = 32;

typedef struct {
    uint32_t lower_freq;
    uint32_t upper_freq;
    uint32_t chirp_rate;
    uint32_t chirps_per_burst;
    uint32_t samples_per_chirp;
} soli_profile;

soli_profile profile_default { 0, 0, 0, 16, 64 };
soli_profile profile_short_range { 58000, 63500, 2000, 64, 16 };
soli_profile profile_long_range { 59000, 61000, 2000, 16, 128 };

bool done = false;
bool delete_output = false;

std::ofstream logfile;
unsigned timeout_secs;
unsigned num_bursts;

struct timeval log_start_time;
struct timeval log_current_time;

bool wait_for_network = false;
bool use_realtime = true;

// return the difference in milliseconds between a pair of timeval
// structures (as returned by gettimeofday)
double timeval_diff_ms(struct timeval& after, struct timeval& before) {
    double diff = 1000 * (after.tv_sec - before.tv_sec);
    return diff + ((after.tv_usec - before.tv_usec) / 1000.0);
}

void BurstDataCallback(const SoliRadarBurst& burst) {
    // burst.is_valid => normally always 1, ignore all data if 0
    // burst.burst_id => index for burst, monotonically increases
    // burst.timestamp_ms => timestamp of the burst in milliseconds
    //
    // each time this callback is triggered, we're receiving the
    // results of 1 burst, consisting of 1 or more chirps, each of 
    // which in turn consists of some number of samples on each of
    // 1-3 (usually always 3) channels

    // log format: each burst will generate the following binary data
    //  - a relative timestamp in milliseconds from the start of the log
    //  - the burst_id and burst_timestamp_ms fields 
    //  - all samples from channel 1
    //  - ...
    //  - all samples from channel n (usually 3)

    gettimeofday(&log_current_time, NULL);
    double timestamp = log_current_time.tv_sec + (log_current_time.tv_usec / 1e6);

    logfile.write(reinterpret_cast<const char*>(&timestamp), sizeof(double));
    logfile.write(reinterpret_cast<const char*>(&(burst.burst_id)), sizeof(uint32_t));
    logfile.write(reinterpret_cast<const char*>(&(burst.timestamp_ms)), sizeof(uint32_t));

    for(const SoliRadarChirp& chirp_data : burst.chirp_data) {
        // write all samples for each channel in turn
        for(size_t i=0;i<chirp_data.channel_data.size();i++)
            logfile.write(reinterpret_cast<const char*>(&(chirp_data.channel_data[i])[0]), chirp_data.channel_data[i].size() * sizeof(float));
    }

    num_bursts++;
    if(num_bursts % 1000 == 0)
        std::cout << "Bursts: " << num_bursts;
}

void signalHandler(int signum) {
    done = true;
    if(signum == SIGINT || signum == SIGKILL)
        delete_output = true;
}

void print_help(void) {
    std::cout << "simplelogger [options] -f logfile\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "\t-t <seconds>\tAuto-exit after this amount of time (otherwise Ctrl-C)\n";
    std::cout << "\t-f <logfile>\tSet the name of the output file\n";
    std::cout << "\t-w If set, wait for any UDP packet on port 12345 before logging starts\n";
    std::cout << "\t-r Use a realtime scheduling mode and increased process priority (default)\n";
    std::cout << "\t-n Do NOT use realtime scheduling mode and increased process priority\n";
    std::cout << "\t-p <profile> Select a profile (default/shortrange/longrange)\n";
    std::cout << "\t-h\tPrint this message\n";
}

bool SetConfig(Soli& soli, const std::string& key, uint32_t value) {
    bool ret = soli.SetConfig(key, value);
    if(!ret)
        std::cout << "Error: failed to set " << key << " to " << value << "!\n";
    return ret;
}

bool ApplyProfile(Soli& soli, const soli_profile& profile) {
    bool ret = true;
    
    std::cout << "ApplyProfile:\n";
    std::cout << "\tLower freq: " << profile.lower_freq << "\n";
    std::cout << "\tUpper freq: " << profile.upper_freq << "\n";
    std::cout << "\tChirp rate: " << profile.chirp_rate << "\n";
    std::cout << "\tChirps per burst: " << profile.chirps_per_burst << "\n";
    std::cout << "\tSamples per chirp: " << profile.samples_per_chirp << "\n";
    std::cout << "\n";

    if(profile.lower_freq > 0)
        ret &= SetConfig(soli, "lower_freq", profile.lower_freq);
    if(profile.upper_freq > 0)
        ret &= SetConfig(soli, "upper_freq", profile.upper_freq);
    if(profile.chirp_rate > 0)
        ret &= SetConfig(soli, "chirp_rate", profile.chirp_rate);
    ret &= SetConfig(soli, "chirps_per_burst", profile.chirps_per_burst);
    ret &= SetConfig(soli, "samples_per_chirp", profile.samples_per_chirp);
    return ret;
}

void realtime_priority(int priority, int schedtype, int niceval)
{
    struct sched_param schedparm;
    memset(&schedparm, 0, sizeof(schedparm));
    schedparm.sched_priority = priority; 
    sched_setscheduler(0, schedtype, &schedparm);
    setpriority(PRIO_PROCESS, 0, niceval);
}

int main(int argc, char *argv[]) {
    uint32_t num_channels = 3;
    uint8_t profile = PROFILE_DEFAULT;
    soli_profile profile_settings = profile_default;
    char* filename = NULL;

    while(1) {
        int c = getopt(argc, argv, "t:f:hwp:rn");
        if(c == -1)
            break;

        switch(c) {
            case 'h':
                print_help();
                return 0;
            case 'f':
                filename = strdup(optarg);
                break;
            case 't':
                timeout_secs = atoi(optarg);
                break;
            case 'w':
                wait_for_network = true;
                break;
            case 'r':
                use_realtime = true;
                break;
            case 'n':
                use_realtime = false;
                break;
            case 'p':
                if(strcmp("shortrange", optarg) == 0) {
                    profile = PROFILE_SHORT_RANGE;
                    std::cout << "Using SHORT RANGE profile\n";
                    profile_settings = profile_short_range;
                } else if (strcmp("longrange", optarg) == 0) {
                    profile = PROFILE_LONG_RANGE;
                    std::cout << "Using LONG RANGE profile\n";
                    profile_settings = profile_long_range;
                } else {
                    std::cout << "Using DEFAULT profile\n";
                }
                break;
        }
    }

    if(filename == NULL) {
        std::cout << "Error: must supply a filename with -f parameter\n";
        return -1;
    }

    if(use_realtime) {
        std::cout << "Using real-time priority settings!";
        realtime_priority(50, SCHED_RR, -10);
    }

    Soli soli;

    // Initialize the radar sensor with the default configuration.
    soli.Init();

    switch(profile) {
        case PROFILE_DEFAULT:
            ApplyProfile(soli, profile_default);
            break;
        case PROFILE_SHORT_RANGE:
            ApplyProfile(soli, profile_short_range);
            break;
        case PROFILE_LONG_RANGE:
            ApplyProfile(soli, profile_long_range);
            break;
    }

    // Reset the sensor as we have changed the config.
    soli.Reset();

    std::cout << "Device configured. ";
    if(timeout_secs > 0) 
        std::cout << "Logging will stop automatically after " << timeout_secs << " seconds\n";
    else
        std::cout << "Use Ctrl-C to stop logging\n";

    gettimeofday(&log_start_time, NULL);
    std::string full_filename(filename);
    switch(profile) {
        case PROFILE_DEFAULT:
            full_filename.append("_default_");
            break;
        case PROFILE_SHORT_RANGE:
            full_filename.append("_shortrange_");
            break;
        case PROFILE_LONG_RANGE:
            full_filename.append("_longrange_");
            break;
    }
    time_t rawtime;
    struct tm* timeinfo;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    char buf[64];
    strftime(buf, 64, "%d%m%Y_%H%M%S", timeinfo);
    full_filename.append(buf);
    full_filename.append(".radar");
    logfile.open(full_filename, std::ios::out | std::ios::trunc | std::ios::binary);
    std::cout << "Logging data to: " << full_filename << "\n";
    free(filename);

    // logfile header. early recordings will just have 3x uint32_t here (chirps_per_burst,
    // samples_per_chirp, num_channels). current version uses a 'soli' marker followed by
    // header length (uint16_t), and then a single byte version field. after this we have:
    // num channels, profile ID, profile fields
    if(false) {
        // old style header for reference
        /* logfile.write(reinterpret_cast<char*>(&chirps_per_burst), sizeof(uint32_t)); */
        /* logfile.write(reinterpret_cast<char*>(&samples_per_chirp), sizeof(uint32_t)); */
        /* logfile.write(reinterpret_cast<char*>(&num_channels), sizeof(uint32_t)); */
    } else {
        logfile.write("soli", 4);
        logfile.write(reinterpret_cast<char*>(&HEADER_SIZE), sizeof(HEADER_SIZE));
        logfile.write(reinterpret_cast<char*>(&HEADER_VERSION), sizeof(HEADER_VERSION));
        logfile.write(reinterpret_cast<char*>(&num_channels), sizeof(num_channels));
        logfile.write(reinterpret_cast<char*>(&profile), sizeof(profile));
        logfile.write(reinterpret_cast<char*>(&profile_settings.lower_freq), sizeof(uint32_t));
        logfile.write(reinterpret_cast<char*>(&profile_settings.upper_freq), sizeof(uint32_t));
        logfile.write(reinterpret_cast<char*>(&profile_settings.chirp_rate), sizeof(uint32_t));
        logfile.write(reinterpret_cast<char*>(&profile_settings.chirps_per_burst), sizeof(uint32_t));
        logfile.write(reinterpret_cast<char*>(&profile_settings.samples_per_chirp), sizeof(uint32_t));
    }

    soli.RegisterBurstCallback(&BurstDataCallback);

    // this is currently used to allow the OptiTrack logger to 
    // trigger logging on the machine with the Soli attached
    int sock = -1;
    struct sockaddr_in sender;
    socklen_t sender_len = sizeof(sender);
    if(wait_for_network) {
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        struct addrinfo *res;
        getaddrinfo("0.0.0.0", "12345", NULL, &res);
        bind(sock, res->ai_addr, res->ai_addrlen);
        std::cout << "Waiting for network to trigger logging...\n";
        int recv = recvfrom(sock, buf, 20, 0, (struct sockaddr*)&sender, &sender_len);
        std::cout << "Received packet of " << recv << " bytes, logging will start now!\n";
        int flags = fcntl(sock, F_GETFL, 0);
        flags |= O_NONBLOCK;
        fcntl(sock, F_SETFL, flags);
    }

    signal(SIGINT, signalHandler);

    // Start streaming radar data, this will trigger the burst callbacks.
    soli.Start();

    double elapsed_time = 0;
    while(!done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        elapsed_time += 0.050;
        if(timeout_secs > 0 && elapsed_time > timeout_secs) {
            done = true;
            std::cout << "Time limit reached, exiting!\n";
        }

        if(sock != -1) {
            // check if logging aborted (loss of marker or other problem)
            int ret = recvfrom(sock, buf, 20, 0, (struct sockaddr*)&sender, &sender_len);
            if(ret == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                // nothing to receive
            } else {
                // abort logging
                std::cout << "*** Received abort packet! Logging will stop\n";
                done = true;
                delete_output = true;
            }
        }
    }

    std::cout << "Closing device\n";

    soli.Stop();

    logfile.close();

    // if interrupted then delete the output file
    if(timeout_secs > 0 && delete_output) {
        std::cout << "Interrupted: deleting output file!\n";
        unlink(full_filename.c_str());
    }

    return 0;
}
