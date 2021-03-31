import win32gui
from pyautogui import screenshot
from datetime import datetime
from time import sleep

def screenshot_window(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd:
        # win32gui.SetForegroundWindow(hwnd)
        if hwnd == win32gui.GetForegroundWindow():
            x, y, x1, y1 = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
            im = screenshot(region=(x, y, x1, y1))
            return im
        else:
            print('Window not visible')
    else:
        print('Window not found!')


while True:
    sleep(5)
    im = screenshot_window('Destiny 2')
    if im:
        im = im.resize((640, 360))
        im.save("screenshots/img_{}.png".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        #im.save('screenshots/test.png')
