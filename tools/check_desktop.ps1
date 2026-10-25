Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class VDCheck {
    [ComImport, Guid("A5CD92FF-29BE-454C-8F04-D82879FB3D34"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IVirtualDesktopManager {
        int IsWindowOnCurrentDesktop(IntPtr topLevelWindow, out int onCurrentDesktop);
        int GetWindowDesktopId(IntPtr topLevelWindow, out Guid desktopId);
        int MoveWindowToDesktop(IntPtr topLevelWindow, ref Guid desktopId);
    }
    public static string Check(int hwnd) {
        Type t = Type.GetTypeFromCLSID(new Guid("AA509086-5CA9-4C25-8F95-589D3C07B48A"));
        IVirtualDesktopManager vdm = (IVirtualDesktopManager)Activator.CreateInstance(t);
        int onCur;
        int hr = vdm.IsWindowOnCurrentDesktop(new IntPtr(hwnd), out onCur);
        Guid gid;
        int hr2 = vdm.GetWindowDesktopId(new IntPtr(hwnd), out gid);
        return string.Format("IsWindowOnCurrentDesktop hr={0} onCurrent={1}; GetWindowDesktopId hr={2} id={3}", hr, onCur, hr2, gid);
    }
}
'@
[VDCheck]::Check(3423766)
