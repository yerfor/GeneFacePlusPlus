import os, re
def clean_gpu():
    ret = os.popen("fuser -v /dev/nvidia*").read()
    ret = re.sub("kernel", " ", ret)
    ids = set(ret.split(" "))
    ids = [int(i) for i in ids if i != '']
    ids = [str(i) for i in sorted(ids)]
    ids_string = ' '.join(ids)
    cmd = f"kill -9 {ids_string}"
    os.system("fuser -v /dev/nvidia*")
    flag = input(f"You are going run this command: \n  ==>  \"{cmd}\" \nEnter y/Y to proceed, or other to abort.\n[y/n]")
    if flag.lower() == 'y':
        os.system(cmd)
        print("All gpu process cleaned!")
    else:
        print("Aborted!")

if __name__ == '__main__':
    clean_gpu()