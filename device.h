#pragma once
#include <windows.h>
#include <string>
#include <stdexcept>
#include <cstdint>
#include "device_symbols.h"

class DeviceController {
private:
    typedef int  (*FnInitDevice)();
    typedef void (*FnReleaseDevice)();
    typedef int  (*FnSendAbsolute)(unsigned short x, unsigned short y);
    typedef int  (*FnSendDelta)(short dx, short dy);
    typedef int  (*FnSendDeltaStep)(short dx, short dy, unsigned char delay, unsigned char delta);
    typedef void (*FnApplyPressCode)(const char* code);
    typedef void (*FnApplyReleaseCode)(const char* code);
    typedef void (*FnApplyTapCode)(const char* code, int min, int max);
    typedef int  (*FnQueryCodeState)(int vk);
    typedef int  (*FnApplyButtonPress)(unsigned char index);
    typedef int  (*FnApplyButtonRelease)(unsigned char index);
    typedef int  (*FnApplyButtonTap)(unsigned char index, int min, int max);
    typedef void (*FnRestartUnit)();
    typedef void (*FnClearAllPress)();
    typedef void (*FnApplyCipher)(unsigned char* data);
    typedef long (*FnQueryUnitId)();
    typedef int  (*FnQueryBuildNum)();
    typedef int  (*FnQueryUnitType)();
    typedef void (*FnSetSyncMode)(int wait);
    typedef int  (*FnCheckActive)();
    typedef int  (*FnInitById)(int id1, int id2);
    typedef void (*FnSetBatchInterval)(unsigned char interval);

    HMODULE              hModule;
    FnInitDevice         fnInit;
    FnReleaseDevice      fnRelease;
    FnSendAbsolute       fnSendAbs;
    FnSendDelta          fnSendDelta;
    FnSendDeltaStep      fnSendDeltaStep;
    FnApplyPressCode     fnPressCode;
    FnApplyReleaseCode   fnReleaseCode;
    FnApplyTapCode       fnTapCode;
    FnQueryCodeState     fnQueryCode;
    FnApplyButtonPress   fnBtnPress;
    FnApplyButtonRelease fnBtnRelease;
    FnApplyButtonTap     fnBtnTap;
    FnRestartUnit        fnRestart;
    FnClearAllPress      fnClearAll;
    FnApplyCipher        fnCipher;
    FnQueryUnitId        fnUnitId;
    FnQueryBuildNum      fnBuildNum;
    FnQueryUnitType      fnUnitType;
    FnSetSyncMode        fnSyncMode;
    FnCheckActive        fnCheckActive;
    FnInitById           fnInitById;
    FnSetBatchInterval   fnBatchInterval;

    bool loadFunctions() {
        fnInit          = reinterpret_cast<FnInitDevice>       (GetProcAddress(hModule, SYM_INIT));
        fnRelease       = reinterpret_cast<FnReleaseDevice>    (GetProcAddress(hModule, SYM_RELEASE));
        fnSendAbs       = reinterpret_cast<FnSendAbsolute>     (GetProcAddress(hModule, SYM_SEND_ABS));
        fnSendDelta     = reinterpret_cast<FnSendDelta>        (GetProcAddress(hModule, SYM_SEND_DELTA));
        fnSendDeltaStep = reinterpret_cast<FnSendDeltaStep>    (GetProcAddress(hModule, SYM_SEND_DELTA_STEP));
        fnPressCode     = reinterpret_cast<FnApplyPressCode>   (GetProcAddress(hModule, SYM_PRESS_CODE));
        fnReleaseCode   = reinterpret_cast<FnApplyReleaseCode> (GetProcAddress(hModule, SYM_RELEASE_CODE));
        fnTapCode       = reinterpret_cast<FnApplyTapCode>     (GetProcAddress(hModule, SYM_TAP_CODE));
        fnQueryCode     = reinterpret_cast<FnQueryCodeState>   (GetProcAddress(hModule, SYM_QUERY_STATE));
        fnBtnPress      = reinterpret_cast<FnApplyButtonPress> (GetProcAddress(hModule, SYM_BTN_PRESS));
        fnBtnRelease    = reinterpret_cast<FnApplyButtonRelease>(GetProcAddress(hModule, SYM_BTN_RELEASE));
        fnBtnTap        = reinterpret_cast<FnApplyButtonTap>   (GetProcAddress(hModule, SYM_BTN_TAP));
        fnRestart       = reinterpret_cast<FnRestartUnit>      (GetProcAddress(hModule, SYM_RESTART));
        fnClearAll      = reinterpret_cast<FnClearAllPress>    (GetProcAddress(hModule, SYM_CLEAR_ALL));
        fnCipher        = reinterpret_cast<FnApplyCipher>      (GetProcAddress(hModule, SYM_CIPHER));
        fnUnitId        = reinterpret_cast<FnQueryUnitId>      (GetProcAddress(hModule, SYM_UNIT_ID));
        fnBuildNum      = reinterpret_cast<FnQueryBuildNum>    (GetProcAddress(hModule, SYM_BUILD_NUM));
        fnUnitType      = reinterpret_cast<FnQueryUnitType>    (GetProcAddress(hModule, SYM_UNIT_TYPE));
        fnSyncMode      = reinterpret_cast<FnSetSyncMode>      (GetProcAddress(hModule, SYM_SYNC_MODE));
        fnCheckActive   = reinterpret_cast<FnCheckActive>      (GetProcAddress(hModule, SYM_CHECK_ACTIVE));
        fnInitById      = reinterpret_cast<FnInitById>         (GetProcAddress(hModule, SYM_INIT_BY_ID));
        fnBatchInterval = reinterpret_cast<FnSetBatchInterval> (GetProcAddress(hModule, SYM_BATCH_INTERVAL));

        return fnInit && fnRelease && fnSendAbs && fnSendDelta && fnQueryCode && fnBatchInterval;
    }

public:
    // dllPath: device.dll 的完整路徑，例如 "C:\\path\\to\\device.dll"
    explicit DeviceController(const std::string& dllPath) : hModule(nullptr) {
        hModule = LoadLibraryA(dllPath.c_str());
        if (!hModule) {
            DWORD error = GetLastError();
            throw std::runtime_error("Failed to load module. Error code: " + std::to_string(error));
        }

        if (!loadFunctions()) {
            FreeLibrary(hModule);
            hModule = nullptr;
            throw std::runtime_error("Failed to resolve required exports.");
        }
    }

    ~DeviceController() {
        if (hModule) {
            fnRelease();
            FreeLibrary(hModule);
        }
    }

    // 禁止複製
    DeviceController(const DeviceController&) = delete;
    DeviceController& operator=(const DeviceController&) = delete;

    bool initialize(int id1 = 0, int id2 = 0) {
        if (!fnCheckActive()) {
            return fnInitById(id1, id2) == 1;
        }
        return true;
    }

    void shutdown() {
        if (fnCheckActive()) {
            fnRelease();
        }
    }

    bool isReady() const {
        return fnCheckActive() == 1;
    }

    // 絕對位置移動
    bool sendAbsolute(unsigned short x, unsigned short y) {
        return fnSendAbs(x, y) == 1;
    }

    // 相對移動: dx>0 右, dx<0 左, dy>0 下, dy<0 上
    bool sendDelta(short dx, short dy) {
        return fnSendDelta(dx, dy) == 1;
    }

    // 相對移動 (漸進): dx>0 右, dx<0 左, dy>0 下, dy<0 上
    bool applyDeltaGradual(short dx, short dy, unsigned char delay = 2, unsigned char delta = 5) {
        return fnSendDeltaStep(dx, dy, delay, delta) == 1;
    }

    bool applyPress(unsigned char index) {
        return fnBtnPress(index) == 1;
    }

    bool applyRelease(unsigned char index) {
        return fnBtnRelease(index) == 1;
    }

    bool applyTap(unsigned char index, int minDelay = 0, int maxDelay = 0) {
        return fnBtnTap(index, minDelay, maxDelay) == 1;
    }

    void applyCodePress(const std::string& code) {
        fnPressCode(code.c_str());
    }

    void applyCodeRelease(const std::string& code) {
        fnReleaseCode(code.c_str());
    }

    void applyCodeTap(const std::string& code, int minDelay = 0, int maxDelay = 0) {
        fnTapCode(code.c_str(), minDelay, maxDelay);
    }

    int queryState(int vk) {
        return fnQueryCode(vk);
    }

    long queryUnitId() {
        if (!fnCheckActive() || !fnUnitId) return -1;
        try { return fnUnitId(); } catch (...) { return -1; }
    }

    int queryBuildNum() {
        if (!fnCheckActive() || !fnBuildNum) return -1;
        try { return fnBuildNum(); } catch (...) { return -1; }
    }

    int queryUnitType() {
        if (!fnCheckActive() || !fnUnitType) return -1;
        try { return fnUnitType(); } catch (...) { return -1; }
    }

    void restartUnit() {
        fnRestart();
    }

    void clearAllActive() {
        fnClearAll();
    }

    void applyCipher(unsigned char* data) {
        fnCipher(data);
    }

    void setSyncMode(bool wait) {
        fnSyncMode(wait ? 1 : 0);
    }

    bool lockInput(unsigned char option) {
        if (!hModule) return false;
        typedef int (*FnLockInput)(unsigned char);
        FnLockInput fn = reinterpret_cast<FnLockInput>(GetProcAddress(hModule, SYM_LOCK_INPUT));
        if (!fn) return false;
        return fn(option) == 1;
    }

    bool releaseCodeById(uint8_t code) {
        if (!hModule) return false;
        typedef int (*FnReleaseById)(uint8_t);
        FnReleaseById fn = reinterpret_cast<FnReleaseById>(GetProcAddress(hModule, SYM_RELEASE_BY_ID));
        if (!fn) return false;
        return fn(code) == 1;
    }

    bool lockCursor(unsigned char option) {
        if (!hModule) return false;
        typedef int (*FnLockCursor)(unsigned char);
        FnLockCursor fn = reinterpret_cast<FnLockCursor>(GetProcAddress(hModule, SYM_LOCK_CURSOR));
        if (!fn) return false;
        return fn(option) == 1;
    }

    bool setBatchInterval(unsigned char interval) {
        if (!hModule || !fnBatchInterval) return false;
        try {
            fnBatchInterval(interval);
            return true;
        } catch (...) {
            return false;
        }
    }
};
