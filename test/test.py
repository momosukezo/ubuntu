import numpy as np
# ファイルオープン
def open_file(txt):
   f = open(txt)
   a = []
   for line in f:
      line     = line.rstrip('\n')  # 改行コード削除
      line_sep = line.split(':')    # コロン削除
      a.append(line_sep)
   return a
# メイン処理
def main_loop(txt):
   a   = []
   Sum = ''
   for i in range(len(txt)-1):
      if int(txt[-1][0]) % int(txt[i][0]) == 0:          # 割り切れた場合
         a.append(txt[i])
         a_np   = np.array(a)
         a_sort = sorted(a_np,key=lambda x: int(x[0]))
         out    = np.array(a_sort)[:,1]
   for i in range(len(out)):
      Sum = Sum + out[i]
   if a == []:                                           # 割り切れない場合
      b = txt[-1][0]
      return b
   return Sum

inputfile = open_file('input.txt')
output    = main_loop(inputfile)
print(output)
