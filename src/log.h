#pragma once

#include <stdio.h>
#include <assert.h>
#include <time.h>

enum class Log_Verbosity 
{
    Error = 0,
    Warning, 
    Debug,
    Info
};

enum class Log_Color
{
    Red = 0,
    Green = 1,
    Yellow = 2,
    Blue = 3,
    Purple = 4
};

#define MAX_LOG_LENGTH 8192

#define LOG_ERROR(...) _log(Log_Verbosity::Error, __VA_ARGS__)
#define LOG_WARNING(...) _log(Log_Verbosity::Warning, __VA_ARGS__)
#define LOG_DEBUG(...) _log(Log_Verbosity::Debug, __VA_ARGS__)
#define LOG_INFO(...) _log(Log_Verbosity::Info, __VA_ARGS__)


template <typename... Args>
void _log(Log_Verbosity verbosity, const char* msg, Args... args)
{
    static const char* color_strings[] = {
        "\x1B[31m", // Red
        "\x1B[32m", // Green
        "\x1B[33m", // Yellow
        "\x1B[34m", // Blue
        "\x1B[35m", // Purple
    };

    static Log_Color log_colors[] = {
        Log_Color::Red,
        Log_Color::Yellow,
        Log_Color::Blue,
        Log_Color::Green,
    };

    time_t curr_time;
    struct tm curr_tm;
    time(&curr_time);
#if _WIN32
    errno_t err = localtime_s(&curr_tm, &curr_time);
    assert(err == 0);
    (void)err;
#else
    (void)localtime_r(&curr_time, &curr_tm);
#endif

    char time_string[128];
    strftime(time_string, sizeof(time_string), "[%Y/%m/%d %T] ", &curr_tm);

    char message_str[MAX_LOG_LENGTH];
#if _WIN32
    sprintf_s(message_str, msg, args...);
#else
    //snprintf(message_str, sizeof(message_str), "%s", msg);
    snprintf(message_str, sizeof(message_str), msg, args...);
#endif

    char out_string[MAX_LOG_LENGTH];
    const char* color_ending = "\033[0m";
#if _WIN32
    sprintf_s(out_string, "%s%s%s%s", color_strings[(int)log_colors[(int)verbosity]], time_string, message_str, color_ending);
#else
    snprintf(out_string, sizeof(out_string), "%s%s%s%s", color_strings[(int)log_colors[(int)verbosity]], time_string, message_str, color_ending);
#endif
    puts(out_string);
}