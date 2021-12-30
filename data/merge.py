txt1 = 'list/DISFA_part1_biocular.txt'
txt2 = 'list/DISFA_part3_biocular.txt'
txt3 = 'list/DISFA_combine_1_3_biocular.txt'

lines1 = open(txt1, 'r').readlines()
lines2 = open(txt2, 'r').readlines()

lines = lines1 + '\n' + lines2

file3 = open(txt3, 'w')
file3.writelines(lines)
file3.close()
