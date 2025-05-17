
import datetime
import time


def check_time_window(interval_minutes: int, check_seconds: int = 3) -> bool:
    """
    判断当前时间是否处于每`interval_minutes`分钟的前`check_seconds`秒内

    Args:
        interval_minutes: 时间窗口间隔（分钟），支持1、5、10等正整数
        check_seconds: 需要判断的前N秒区间（默认3秒）

    Returns:
        bool: 当前时间是否符合条件
    """
    now = datetime.datetime.now()

    # 计算当前时间在时间窗口内的秒数偏移
    total_seconds = now.minute * 60 + now.second
    window_seconds = interval_minutes * 60
    offset_seconds = total_seconds % window_seconds  # 当前窗口内的秒数偏移

    return offset_seconds < check_seconds  # 判断是否在前check_seconds秒内

if __name__ == '__main__':
    for i in range(1000):


        print('每1分钟的前3秒-----',check_time_window(1,3))
        print('每2分钟的前3秒-----',check_time_window(2,3))
        print('每3分钟的前3秒-----',check_time_window(3,3))
        print('每4分钟的前3秒-----',check_time_window(4,3))
        print('每5分钟的前3秒-----',check_time_window(5,3))
        time.sleep(3)
