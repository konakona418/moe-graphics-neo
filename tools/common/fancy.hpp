#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>


namespace fancy {
    namespace colors {
        const std::string RESET = "\033[0m";

        const std::string RED = "\x1b[38;2;255;85;85m";
        const std::string GREEN = "\x1b[38;2;80;250;123m";
        const std::string YELLOW = "\x1b[38;2;241;250;140m";
        const std::string BLUE = "\x1b[38;2;139;233;253m";
        const std::string MAGENTA = "\x1b[38;2;255;121;198m";
        const std::string CYAN = "\x1b[38;2;0;255;255m";
        const std::string WHITE = "\x1b[38;2;245;245;245m";
    }// namespace colors

    static inline size_t visibleLength(const std::string& s) {
        size_t count = 0;
        bool in_escape = false;
        for (size_t i = 0; i < s.size(); ++i) {
            if (!in_escape) {
                if (s[i] == '\x1b') {
                    in_escape = true;
                } else {
                    count++;
                }
            } else {
                if (s[i] == 'm') {
                    in_escape = false;
                }
            }
        }
        return count;
    }

    static inline void appendRainbowChar(std::ostringstream& o, char c, float offset) {
        float r = std::sin(offset + 0.0f) * 127 + 128;
        float g = std::sin(offset + 2.0f) * 127 + 128;
        float b = std::sin(offset + 4.0f) * 127 + 128;

        o << "\x1b[38;2;" << (int) r << ";" << (int) g << ";" << (int) b << "m" << c << "\x1b[0m";
    }

    static inline void progressBar(float progress, const std::string& info) {
        static size_t last_visible = 0;

        int width = 50;
        int pos = progress * width;

        std::ostringstream oss;
        oss << "\r[";

        const float freq = 0.10f;
        for (int i = 0; i < width; ++i) {
            float offset = freq * i;
            char c = (i < pos ? '=' : (i == pos ? '>' : ' '));
            appendRainbowChar(oss, c, offset);
        }

        oss << "] ";

        oss << std::fixed << std::setprecision(1) << (progress * 100.0f) << "% ";
        oss << info;

        std::string out = oss.str();

        size_t vis = visibleLength(out);
        if (vis < last_visible) {
            out.append(last_visible - vis, ' ');
        }

        last_visible = vis;

        std::cout << out << std::flush;
    }

    static inline void printRainbow(const std::string& text) {
        const float freq = 0.3f;
        for (size_t i = 0; i < text.size(); ++i) {
            float r = std::sin(freq * i + 0.0f) * 127 + 128;
            float g = std::sin(freq * i + 2.0f) * 127 + 128;
            float b = std::sin(freq * i + 4.0f) * 127 + 128;

            std::cout << "\x1b[38;2;"
                      << (int) r << ";" << (int) g << ";" << (int) b << "m"
                      << text[i];
        }
        std::cout << "\x1b[0m";
    }
}// namespace fancy