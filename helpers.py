import os


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return float(convert_bytes(file_info.st_size).split(' ')[0])

def find_all_tifs(root_dir, fsize=5):
    file_list = []
    for dirname, subdirlist, filelist in os.walk(root_dir):
        for f in filelist:
            if 'DeepBlue' in f:
                continue
            if '.tif' in f:
                full_path = os.path.join(dirname, f)
                if file_size(full_path)>5:
                    file_list.append(full_path)
    return file_list


