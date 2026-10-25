"""用 ctypes 直接调 IVirtualDesktopManager COM，检查游戏窗口是否在当前虚拟桌面。"""
import ctypes
from ctypes import byref, c_int, c_long, c_ubyte, c_ulong, c_ushort, c_void_p, c_wchar_p

ole32 = ctypes.OleDLL("ole32")


class GUID(ctypes.Structure):
    _fields_ = [("Data1", c_ulong), ("Data2", c_ushort),
                ("Data3", c_ushort), ("Data4", c_ubyte * 8)]


def guid(s: str) -> GUID:
    g = GUID()
    ole32.CLSIDFromString(c_wchar_p(s), byref(g))
    return g


CLSID = guid("{AA509086-5CA9-4C25-8F95-589D3C07B48A}")
IID = guid("{A5CD92FF-29BE-454C-8F04-D82879FB3D34}")
CLSCTX_ALL = 23

hr = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED
print(f"CoInitializeEx hr={hr:#x}")

pv = c_void_p()
hr = ole32.CoCreateInstance(byref(CLSID), None, CLSCTX_ALL, byref(IID), byref(pv))
print(f"CoCreateInstance hr={hr:#x} ptr={pv.value:#x}" if pv.value else f"CoCreateInstance hr={hr:#x} (failed)")

if pv.value:
    vtable = ctypes.cast(pv.value, ctypes.POINTER(c_void_p)).contents
    # 找 vtable 起始：对象首字段即 vtable 指针
    obj = ctypes.cast(pv.value, ctypes.POINTER(c_void_p))
    vt = ctypes.cast(obj[0], ctypes.POINTER(c_void_p))
    FN = ctypes.WINFUNCTYPE(c_long, c_void_p, c_void_p, ctypes.POINTER(c_int))
    is_on_current = FN(vt[3])
    hwnd = c_void_p(3423766)
    on_cur = c_int(-1)
    hr = is_on_current(pv, hwnd, byref(on_cur))
    print(f"IsWindowOnCurrentDesktop hr={hr:#x} onCurrent={on_cur.value}")
