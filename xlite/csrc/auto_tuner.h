#ifndef XLITE_AUTO_TUNER_H
#define XLITE_AUTO_TUNER_H

inline uint32_t GetTileSizeOfCachedKV(uint32_t aicNum)
{
    if (aicNum == 20) {
        return 8192;
    } else if (aicNum == 24) {
        return 6016;
    }
    return 8192;
}

#endif