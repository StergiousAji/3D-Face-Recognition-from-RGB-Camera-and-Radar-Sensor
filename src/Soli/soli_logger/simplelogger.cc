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

#ifdef ZMQ
#include <zmq.hpp>
#endif

#include "soli.h"

#define PROFILE_DEFAULT         0
#define PROFILE_SHORT_RANGE     1
#define PROFILE_LONG_RANGE      2 
#define PROFILE_LONG_RANGE_V2   3

typedef struct {
    uint32_t lower_freq;
    uint32_t upper_freq;
    uint32_t chirp_rate;
    uint32_t chirps_per_burst;
    uint32_t samples_per_chirp;
} soli_profile;

// this is the default profile the hardware uses if not otherwise configured
soli_profile profile_default       = {0, 0, 0, 16, 64};
// "short range" profile: max sensing range 20cm, range bins 2.7cm
soli_profile profile_short_range   = { 58000, 63500, 2000, 64, 16 };
// "long range" profile: max sensing range 4.8m, range bins 7.5cm
soli_profile profile_long_range    = { 59000, 61000, 2000, 16, 128 };
// "long range v2" profile: max sensing range 9.6m, range bins 15cm
soli_profile profile_long_range_v2 = { 59500, 60500, 2000, 16, 128 };

uint8_t HEADER_VERSION  = 1;
uint16_t HEADER_SIZE    = 32;

bool done               = false;
bool delete_output      = false;

std::ofstream logfile;
unsigned timeout_secs;
unsigned num_bursts;

struct timeval log_start_time;
struct timeval log_current_time;

bool wait_for_network   = false;
bool use_realtime       = true;
bool tcp_streaming      = false;
bool use_zmq            = false;
bool local_logging      = false;
int tcp_server_sock     = -1;
int tcp_client_sock     = -1;
char* burstbuffer       = NULL;
uint32_t burstbuffer_sz = 0;
uint32_t print_bursts   = 1000;

#ifdef ZMQ
zmq::context_t zmq_ctx(1);
zmq::socket_t zmq_pub(zmq_ctx, ZMQ_PUB);
#endif

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

    uint32_t offset = HEADER_SIZE;
    gettimeofday(&log_current_time, NULL);
    double timestamp = log_current_time.tv_sec + (log_current_time.tv_usec / 1e6);

    memcpy(burstbuffer + offset, reinterpret_cast<const char*>(&timestamp), sizeof(double));
    offset += sizeof(double);
    memcpy(burstbuffer + offset, reinterpret_cast<const char*>(&(burst.burst_id)), sizeof(uint32_t));
    offset += sizeof(uint32_t);
    memcpy(burstbuffer + offset, reinterpret_cast<const char*>(&(burst.timestamp_ms)), sizeof(uint32_t));
    offset += sizeof(uint32_t);

    uint32_t chirp_index = 0;
    for(const SoliRadarChirp& chirp_data : burst.chirp_data) {
        uint32_t channel_data_size = sizeof(float) * chirp_data.channel_data[0].size();

        // write all samples for each channel in turn
        for(size_t i=0;i<chirp_data.channel_data.size();i++) {
            memcpy(burstbuffer + offset, reinterpret_cast<const char*>(&(chirp_data.channel_data[i][0])), channel_data_size);
            offset += channel_data_size;
        }

        chirp_index++;
    }

    if(local_logging)
        logfile.write(burstbuffer + HEADER_SIZE, (offset - HEADER_SIZE));

    num_bursts++;
    if(num_bursts % print_bursts == 0)
        std::cout << "Bursts: " << num_bursts << "\n";

    #ifdef ZMQ
    // if ZMQ streaming enabled, publish a message containing each burst we receive
    // with the "soli" topic string
    if(use_zmq) {
        zmq_pub.send(zmq::str_buffer("soli"), zmq::send_flags::sndmore);
        zmq::message_t msg(burstbuffer, offset, NULL, NULL);
        zmq_pub.send(msg, zmq::send_flags::dontwait);
    }
    #endif

    // if TCP streaming mode is active and we don't already have a client connected
    if(tcp_streaming && tcp_client_sock == -1) {

        socklen_t addr_size;
        struct sockaddr_storage remote_addr;
        
        // check for any incoming connections and store the client socket if successful
        int result = accept(tcp_server_sock, (struct sockaddr*)&remote_addr, &addr_size);
        if(result == -1) {
            if(errno == EAGAIN || errno == EWOULDBLOCK) {
                // ignore
            } else {
                // other error
                perror("accept() error: ");
            }
        } else {
            tcp_client_sock = result;
            std::cout << "Client connection made\n";
        }
    }

    // if TCP streaming is active and we have a client connected
    if(tcp_streaming && tcp_client_sock != -1) {
        if(send(tcp_client_sock, burstbuffer, offset, 0) == -1) {
            if(errno == ECONNRESET || errno == EAGAIN || errno == EPIPE) {
                std::cout << "Client disconnected\n";
            } else {
                perror("send() error: ");
            }
            close(tcp_client_sock);
            tcp_client_sock = -1;
        }
    }
}

void signalHandler(int signum) {
    std::cout << "Got signal " << signum << "\n";
    if(signum == SIGPIPE) {
        std::cout << "Caught SIGPIPE, client socket disconnected\n";
        return;
    }
    // exit on SIGKILL/SIGINT
    done = true;
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
    std::cout << "\t-p <profile> Select a profile (default/shortrange/longrange/longrange_v2)\n";
    std::cout << "\t-s Use TCP data streaming (only one of -z/-s can be used at the same time)\n";
    std::cout << "\t-z Use ZMQ data streaming (only one of -z/-s can be used at the same time)\n";
    std::cout << "\t-b <num>\tPrint a message every <num> bursts received\n";
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
#ifndef __APPLE__
    struct sched_param schedparm;
    memset(&schedparm, 0, sizeof(schedparm));
    schedparm.sched_priority = priority; 
    sched_setscheduler(0, schedtype, &schedparm);
#endif
    setpriority(PRIO_PROCESS, 0, niceval);
}

int main(int argc, char *argv[]) {
    uint32_t num_channels = 3;
    uint8_t profile = PROFILE_DEFAULT;
    soli_profile profile_settings = profile_default;
    char* filename = NULL;

    while(1) {
        int c = getopt(argc, argv, "t:f:hwp:rnszb:");
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
            case 's':
                tcp_streaming = true;
                break;
            case 'b':
                print_bursts = atoi(optarg);
                break;
            case 'z':
                #ifdef ZMQ
                    use_zmq = true;
                #else
                    std::cout << "Recompile the application with ZMQ support\n";
                    return -1;
                #endif
                break;
            case 'p':
                if(strcmp("shortrange", optarg) == 0) {
                    profile = PROFILE_SHORT_RANGE;
                    std::cout << "Using SHORT RANGE profile\n";
                    profile_settings = profile_short_range;
                } else if(strcmp("longrange", optarg) == 0) {
                    profile = PROFILE_LONG_RANGE;
                    std::cout << "Using LONG RANGE profile\n";
                    profile_settings = profile_long_range;
                } else if(strcmp("longrange_v2", optarg) == 0) {
                    profile = PROFILE_LONG_RANGE_V2;
                    std::cout << "Using LONG RANGE_V2 profile\n";
                    profile_settings = profile_long_range_v2;
                } else {
                    std::cout << "Using DEFAULT profile\n";
                }
                break;
        }
    }

    if(tcp_streaming && use_zmq) {
        std::cout << "You can only select one of the '-s' and '-z' parameters\n";
        return -1;
    }

    // only require a filename if not streaming
    if(filename == NULL && !tcp_streaming && !use_zmq) {
        std::cout << "Error: must supply a filename with -f parameter\n";
        return -1;
    }

    if(use_realtime) {
        std::cout << "Using real-time priority settings!\n";
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

    std::string full_filename;
    gettimeofday(&log_start_time, NULL);
    char buf[64];

    if(filename != NULL) {
        full_filename = filename;
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
        strftime(buf, 64, "%d%m%Y_%H%M%S", timeinfo);
        full_filename.append(buf);
        full_filename.append(".radar");
        logfile.open(full_filename, std::ios::out | std::ios::trunc | std::ios::binary);
        std::cout << "Logging data to: " << full_filename << "\n";
        free(filename);
        local_logging = true;
    } else {
        local_logging = false;
    }

    // fixed header info
    burstbuffer_sz = 4 + sizeof(HEADER_SIZE) + sizeof(HEADER_VERSION) + sizeof(num_channels) + sizeof(profile) + (5 * sizeof(uint32_t));
    // per-burst fields
    burstbuffer_sz += sizeof(double) + (2 * sizeof(uint32_t));
    // amount of data in each burst: number of chirps in burst * number of samples 
    // per chirp * number of channels (always 3 here) * size of a float
    burstbuffer_sz += profile_settings.chirps_per_burst * profile_settings.samples_per_chirp * 3 * sizeof(float);
    burstbuffer = (char*)malloc(burstbuffer_sz);
    std::cout << "BurstBuffer size = " << burstbuffer_sz << "\n";

    if(tcp_streaming) {
        tcp_server_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        int enable = 1;
        setsockopt(tcp_server_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
        setsockopt(tcp_server_sock, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));

        struct addrinfo *res;
        getaddrinfo("0.0.0.0", "22388", NULL, &res);
        if(bind(tcp_server_sock, res->ai_addr, res->ai_addrlen) == -1) {
            std::cout << "Error: failed to bind port 22388 for streaming data!\n";
            soli.Stop();
            if(local_logging)
                logfile.close();
            return -1;
        }
        if(listen(tcp_server_sock, 1) == -1) {
            std::cout << "Error: failed to listen()\n";
            soli.Stop();
            if(local_logging)
                logfile.close();
            return -1;
        }

        int flags = fcntl(tcp_server_sock, F_GETFL, 0);
        fcntl(tcp_server_sock, F_SETFL, flags |= O_NONBLOCK);

        std::cout << "Waiting for incoming TCP connections\n";
    }

    #ifdef ZMQ
    if(use_zmq) {
        zmq_pub.bind("tcp://*:5556");
        std::cout << "ZMQ publisher socket created\n";
    }
    #endif

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
        strncat(burstbuffer, "soli", 4);
        uint32_t offset = 4;
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&HEADER_SIZE), sizeof(HEADER_SIZE));
        offset += sizeof(HEADER_SIZE);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&HEADER_VERSION), sizeof(HEADER_VERSION));
        offset += sizeof(HEADER_VERSION);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&num_channels), sizeof(num_channels));
        offset += sizeof(num_channels);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&profile), sizeof(profile));
        offset += sizeof(profile);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&profile_settings.lower_freq), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&profile_settings.upper_freq), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&profile_settings.chirp_rate), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&profile_settings.chirps_per_burst), sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(burstbuffer + offset, reinterpret_cast<char*>(&profile_settings.samples_per_chirp), sizeof(uint32_t));
        offset += sizeof(uint32_t);

        logfile.write(burstbuffer, offset);
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
    signal(SIGKILL, signalHandler);
    signal(SIGPIPE, signalHandler);

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
    if(local_logging)
        logfile.close();
    if(burstbuffer != NULL)
        free(burstbuffer);
    if(tcp_streaming) {
        close(tcp_server_sock);
        if(tcp_client_sock != -1)
            close(tcp_client_sock);
    }

    #ifdef ZMQ
    if(use_zmq) {
        zmq_pub.close();
        zmq_ctx.close();
    }
    #endif

    // if interrupted then delete the output file
    if(filename != NULL && timeout_secs > 0 && delete_output) {
        std::cout << "Interrupted: deleting output file!\n";
        unlink(full_filename.c_str());
    }

    return 0;
}
