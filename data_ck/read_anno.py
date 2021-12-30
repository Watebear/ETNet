import os

data_root = '/home/dddzz/workspace/Datasets/CK+'
img_root = '/home/dddzz/workspace/Datasets/CK+/cohn-kanade-images'
fac_root = '/home/dddzz/workspace/Datasets/CK+/FACS'
ldm_root = '/home/dddzz/workspace/Datasets/CK+/Landmarks'

def str2int(inp_str):
    if inp_str[-1] == '1':
        oup = int(inp_str[0] + inp_str[2])
    elif inp_str[-1] == '0':
        oup = int(inp_str[0])
    return oup

def static_ck_plus():
    au_cls = {}
    # static
    for sub in os.listdir(fac_root):
        for seq in os.listdir(os.path.join(fac_root, sub)):
            for fac in os.listdir(os.path.join(fac_root, sub, seq)):
                txt_path = os.path.join(fac_root, sub, seq, fac)
                txt = open(txt_path)
                facs = txt.readlines()
                for fac in facs:
                    try:
                        fac = fac.strip().split()[0]
                        fac = str2int(fac)
                        if str(fac) not in au_cls:
                            au_cls[str(fac)] = 1
                        else:
                            au_cls[str(fac)] += 1
                    except:
                        print("false: ", txt_path)
    # report
    for k, v in au_cls.items():
        print(k, ':', v)


# path au1 au2 au6 au7 au12 au15 au17 au23 au24
def gen_file_list():
    write_txt = open('ck_au_list.txt', 'w')
    keep = {'1': 0, '2': 1, '6': 2, '7': 3, '12': 4, '15': 5, '17': 6, '23': 7, '24': 8}
    for sub in os.listdir(fac_root):
        for seq in os.listdir(os.path.join(fac_root, sub)):
            for fac in os.listdir(os.path.join(fac_root, sub, seq)):
                txt_path = os.path.join(fac_root, sub, seq, fac)
                txt = open(txt_path)
                facs = txt.readlines()

                au_labels = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
                path = "/".join(os.path.join(fac_root, sub, seq, fac).split('/')[7:])
                path = path.replace('_facs.txt', '.png')
                for fac in facs:
                    fac = fac.strip().split()[0]
                    fac = str2int(fac)
                    if str(fac) in keep:
                        au_labels[keep[str(fac)]] = '1'
                to_write = path + " " + " ".join(au_labels)
                write_txt.write(to_write+'\n')
    write_txt.close()

def static_file_list():
    static_txt = open('ck_au_list.txt', 'r')
    keep_inv = {'1': 0, '2': 1, '6': 2, '7': 3, '12': 4, '15': 5, '17': 6, '23': 7, '24': 8}
    num_aus = {'1': 0, '2': 0, '6': 0, '7': 0, '12': 0, '15': 0, '17': 0, '23': 0, '24': 0}
    lines = static_txt.readlines()
    for line in lines:
        line = line.strip().split()
        print(line)
        if line[1] == '1':
            num_aus['1'] += 1
        if line[2] == '1':
            num_aus['2'] += 1
        if line[3] == '1':
            num_aus['6'] += 1
        if line[4] == '1':
            num_aus['7'] += 1
        if line[5] == '1':
            num_aus['12'] += 1
        if line[6] == '1':
            num_aus['15'] += 1
        if line[7] == '1':
            num_aus['17'] += 1
        if line[8] == '1':
            num_aus['23'] += 1
        if line[9] == '1':
            num_aus['24'] += 1
    static_txt.close()
    for k, v in num_aus.items():
        print("au{}={}".format(k, v))




if __name__ == "__main__":
    gen_file_list()
    static_file_list()


