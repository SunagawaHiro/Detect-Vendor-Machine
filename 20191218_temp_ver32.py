# テンプレートマッチング(NCC)のプログラム
# 参照URL = https://algorithm.joho.info/programming/python/opencv-template-matching-ncc-py/
# 域値を設定することで、その域値以上の検出結果を描画するプログラム

# ver1に追記する
# 回転処理を加えたプログラムを記述する(達成)
# このプログラムは、回転しても全体がきちんと映るように調整されています。

# ver2に追記
# 回転画像の黒以外の部分をテンプレートマッチングにかける(未達成)
# Q. 入力画像を回転させ、それに対してテンプレートマッチングを試すことで解決することはできないだろうか？
# A. できた。pythonとOpenCVの関数のそれぞれでマッチングした時の処理結果の違いが少し気になる

# ver4に追記
# OpenCVでマッチテンプレートを実行する処理を追加した
# OpecCVでマッチテンプレートを行うと、pythonのプログラムより早い。全体の処理が1分以内に完了する感じ

# ver5に追記
# 正面画像に限定する方針に切り替えました。それにより,回転処理系の関数などは消すようになっています。
# python のマッチテンプレート関数は削除しました
# 回転させる処理も削除しました
# 回転を行う関数も削除しました
# 描画する関数も削除しました
# カラーヒストグラムを行う関数を追加しました(参照URL https://qiita.com/best_not_best/items/c9497ffb5240622ede01)
# カラーヒストグラムの類似度の結果を表示できるようになりました
# 3511.jpg

# ver6に追記
# 画像の2点を指定し, その直線上のヒストグラムを求める関数を追加しました

# ver7に追記
# グラフのパターンを検出したい

# ver8に追記
# 自己相関関数を使えるようにしました

# ver9に追記
# 2点間を通る直線の方程式(y=m*x+n)がわかるようになりました
# 類似度グラフを表示できるようになりました(座標は画像の座標と対応していません)

# ver10に追記
# 類似度グラフを表示できるようになりました(グラフの横軸座標と画像座標は対応しています)
# 3511の類似度データを他ファイルから読み込めるようになりました. マッチングの計算を行わなくてもよくなりました.
# ピーク検出を行なった類似度グラフを表示できるようになりました.

# ver11に追記
# 4245.jpg の画像に対してこのプログラムを適用しました

# ver12に追記
# 3916.jpg の画像に対してこのプログラムを適用し, 類似度グラフを作成しました.
# 3841.jpg の画像に対してこのプログラムを適用し, 類似度グラフを作成しました.

# ver13に追記
# テンプレートマッチングで検出した個数をグラフにして表示できるようになりました.
# グラフのピーク値から, どこにボタン列があるのかを判定できるようになりました.

# ver14に追記
# 検出ボタンの個数グラフのピーク値(どの座標にボタン列がありそうかが分かる)から,
# 各列における類似度の高いボタンを4つ抜き出すことができるようになりました.
# 検出したボタン列が3列以外(0,1,2列)の場合, プログラムがうまくいかないバグが発見されました.

# ver15に追記
# ver15で発見されたバグを直しました.すなわちピーク(ボタン列)が3列以外だとしても動作するようになりました
# 

# ver16に追記
# スケール変換しながらテンプレマッチングを行なった際,最も検出個数が多かったときのテンプレートのスケール,
# そのスケールで変換を行なった後のテンプレートを取得できるようになりました.
# スケール変換の関数を削除しました.
# ボタン列のピーク値が本当に正しいかどうかを判定する機能を追加しました.

# ver17に追記
# 検出された1つのボタン列のピークについて,回帰直線を引いて, その間のパターンを求めることができました。

# ver18に追記
# 検出された全てのボタン列に対して回帰直線を引いて, ピーク検出を行えるようになりました.

# ver19に追記
# 類似度が域値以上のピークのみを抽出することができるようになりました.

# ver20に追記
# ピークの域値を最大類似度*0.8に設定しました. これで信頼性のあるピークのみを抽出できるようになりました

# ver21に追記
# 検出したピーク間の距離を比較し、平均のボタン距離を求め, 上段のドリンクに関しては,ボトル画像を抽出できるようになりました

# ver22に追記
# 検出した部分の類似度が高い部分を抽出する処理の部分を関数化して, わかりやすくなりました.
# ボトルを切り出して保存する処理を関数化して、よりプログラムが読みやすくなりました.

# ver23に追記
# ver23で関数化したコードの部分の処理をmain()から削除しました.
# 自販機画像とボタン画像を放り込めば,そこから何段映っているかを検出し, ボトル画像を自動で抽出することが可能になりました

# ver24に追記
# 今まで試していない画像にも試しました. プログラムは変わっていません.
# 変更点は読み込むファイル(自販機画像とボタンのテンプレート画像)のみです.
# 同時に複数の画像に対して処理を行いたかったため, 新たにプログラムを作成しました.

# ver25に追記
# 1度のテンプレマッチングのみで, ボトルの切り出しを行えるようになりました.
# 計算時間が40倍程度早くなりました.

# ver26に追記
# どの自販機を対象にするのかをコマンドラインから指定できるようになりました.
# auto_outputディレクトリ に結果を自動で格納できるようになりました.

# ver27に追記
# ボタン列のピークのグラフを "button_peak_graph" に保存できるようになりました.
# 商品の各段のピークを "drink_peak_graph" に保存できるようになりました.
# ドリンク画像のピークの抽出の前後をグラフから見れるようになりました.
# 類似度が最大のボタンを選び出し, それを新しいテンプレートにする処理を加えました.

# ver28に追記
# 「ボタン列の回帰直線を求めてから~」という手法を脱却し, 単純に最初の類似度のみでマップを作ることができないかを試すことに変更しました.
# このver28から仕様が大きく変わりました.
# それぞれのピーク位置からボタンを選ぶ処理を削除しました.

# ver29に追記
# 新しいテンプレートでマッチングを行い,そのさいのピークを使用するという処理を削除しました.
# ボタンを抽出する処理をアップデートしました.ボタンをいくつ抽出するのかが調整しやすくなりました.
# 以前のボタンを抽出する処理(get_button_line_points3など)は全て削除しました.

# ver30に追記
# "# 検出したボタン列のピークの範囲を元に類似度グラフ(縦軸類似度,横軸画像の横幅)を作成する"の部分の処理を削除しました.
# 各ボタン列からボタンを抽出する手法をアップデートしました.抽出するボタン数を設定できるようになりました.

# ver31に追記
# 「~本 x ~段の自販機です. ~ 段目の ~ 番目に ~ があります」を表示できるようにしたい.
#


import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import statsmodels.api as smg
from statsmodels.graphics.tsaplots import plot_acf
import match_data as md # 独自モジュール
import datetime
import time


# カラーヒストグラムを用いた類似度マッチング
def color_hist_compare(temp_img, crop_img):
    FileName = "3841"
    
    # リサイズする関数(スケール変更した際に必要になりそう)
    IMG_SIZE = (191,102)
    temp_img = cv2.resize(temp_img, IMG_SIZE)
    cv2.imwrite("./output/{}_jpg/temp.png".format(FileName), temp_img)
    crop_img = cv2.resize(crop_img, IMG_SIZE)
    cv2.imwrite("./output/{}_jpg/hikaku_taisyou.png".format(FileName), crop_img)

    # テンプレートとクロップ画像のヒストグラムを計算し, 比較する
    target_hist = cv2.calcHist([temp_img], [0], None, [256], [0, 256])    
    comparing_hist = cv2.calcHist([crop_img], [0], None, [256], [0, 256])        
    ret = cv2.compareHist(target_hist, comparing_hist, 0)

    return ret



# 2点間の直線のヒストグラムを求める関数(まだ効果はわからない。これから使えるかどうか検証する)
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]
    
    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)
    
    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)
    
    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]
    
    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


# 移動平均を行う関数
def get_mave(sum_list, radius):
    mave_sum_x = []
    for i in range(radius, len(sum_list)-radius):
        sum_element = 0
        sum_count = 0
        for j in range(-radius, radius+1, 1):
            sum_element += sum_list[i+j]
            sum_count += 1
        mave_sum_x.append(sum_element/sum_count)        
    return mave_sum_x

# 自己相関を求める関数
def autocorrelation(data, k):
    """Returns the autocorrelation of the *k*th lag in a time series data.

    Parameters
    ----------
    data : one dimentional numpy array
    k : the *k*th lag in the time series data (indexing starts at 0)
    """

    # yの平均
    y_avg = np.mean(data)

    # 分子の計算
    sum_of_covariance = 0
    for i in range(k+1, len(data)):
        covariance = ( data[i] - y_avg ) * ( data[i-(k+1)] - y_avg )
        sum_of_covariance += covariance

    # 分母の計算
    sum_of_denominator = 0
    for u in range(len(data)):
        denominator = ( data[u] - y_avg )**2
        sum_of_denominator += denominator

    return sum_of_covariance / sum_of_denominator



# 2点を通る直線の方程式を返す(y=mx+n)
def makeLinearEquation(x1, y1, x2, y2):
	line = {}
	if y1 == y2:
		# y軸に平行な直線
		line["y"] = y1
	elif x1 == x2:
		# x軸に平行な直線
		line["x"] = x1
	else:
		# y = mx + n
		line["m"] = (y1 - y2) / (x1 - x2)
		line["n"] = y1 - (line["m"] * x1)
	return line

# x座標を渡すと, y座標が返ってくる関数
def get_y_crd(m,n,x):
    y = m*x + n
    return y


# 最も検出個数の多いスケールとそのときの画像を取得する
def get_max_button_scale(gray_input,temp):    
    nums_scale = []
    h_original, w_original = temp.shape
    threshold = 0.7
    button_nums = []
    for i in range(5,16,1):
        scale = i/10
        template = cv2.resize(temp, dsize=(int(w_original*scale), int(h_original*scale)))
        res = cv2.matchTemplate(gray_input,template,cv2.TM_CCOEFF_NORMED)    
        loc = np.where( res >= threshold)
        button_nums.append(len(loc[0]))
        
    print("button_nums = ",button_nums)
    print("max(button_nums):",max(button_nums))
    print("button_nums.index(max(button_nums))",button_nums.index(max(button_nums)))
    nums_scale.append(max(button_nums))
    nums_scale.append(button_nums.index(max(button_nums))+5)
        
    scale = nums_scale[1]/10
    template = cv2.resize(temp, dsize=(int(w_original*scale), int(h_original*scale)))
    
    return nums_scale,template

def get_xy_points(choice_xy_sml, w_move, h_move):
    x_points = []
    y_points = []
    
    for i in choice_xy_sml:
        x_points.append(i[0][0]+w_move)
        y_points.append(i[0][1]+h_move)
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    return x_points,y_points

def get_new_line_crd(func,w_input,w_move,h_move,wt):
    line_crd = []
    count = 0
    for x in range(0,w_input,1):
        value = (x,int(func(x)))
        line_crd.append(value)    
        count += 1
        if (count % 500) == 0:
            print("value = ",value)            
            
    print("len(line_crd) = ",len(line_crd))
    print()    
    # 左上の座標に持って行くための移動幅(line_crdをテンプレの中心として、テンプレートを切り抜かなければならない。)
    # そのため, line_crd に格納されている座標から左上に持っていくために テンプレのサイズ(縦横)の半分を知っている必要がある
    print("h_move, w_move",h_move,w_move)
    # 直線の左端と右端を削除する(テンプレートマッチングとして利用しないため)
    print("line_crd の長さ",len(line_crd))
    new_line_crd = line_crd[0:w_input-wt] # スライスの仕方が間違っている?
    print("new_line_crd の長さ",len(new_line_crd))
    return new_line_crd


def get_ruijido_list(new_line_crd,FileName,w_move,h_move,wt,ht,temp):
    # クロップして,matchテンプレート関数にかける
    # 直線上の類似度マップを求める. +のx軸方向に1つずつ座標をずらしながら類似度を計算していく
    ruijido_list = [] # 類似度を格納する配列
    count = 0
    for i in range(len(new_line_crd)):        
        gray_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName),0)
        top = new_line_crd[i][1] - h_move
        bottom = new_line_crd[i][1] + h_move
        left = new_line_crd[i][0] - w_move
        right = new_line_crd[i][0] + w_move
        crop_each = gray_input[ top : bottom, left : right] # img[top : bottom, left : right]        
        # リサイズする
        crop_each = cv2.resize(crop_each , (wt,ht))        
        # テンプレートマッチングを行う
        res = cv2.matchTemplate(crop_each,temp,cv2.TM_CCOEFF_NORMED)        
        ruijido_list.append(res[0][0])
        count += 1        
        if count % 200 == 0:
            print(count)            
    return ruijido_list


def draw_graph(ruijido_list,w_move,w_input,FileName,count):            
    left = np.array(list(range(w_move,w_input-w_move,1)))
    ruijido_list = np.array(ruijido_list)
    print("len(ruijido_list)",len(ruijido_list))    
    plt.bar(left, ruijido_list)
    plt.savefig("./output/{}_jpg/20191129/{}_ruijido_graph_S{}.png".format(FileName,FileName,count))    
    print("{}枚目".format(count))
    plt.close()

    
# 回帰直線上を元にテンプレマッチングを行なった結果である ruijido_listを受け取り、そのリスト内から,ピークを抽出する関数
# ピークを抽出することで、最も類似度が高い場所(そこはおそらくボタンの位置であるはず)の座標を得ることができる
def get_peak_of_ruijido_list(new_data_3841,th,wt,w_move,count,FileName): # 関数化1
    # 検出したボタン列の中でマッチングを行い,ピーク検出を行う
    print()
    data_3841_numpy = np.array(new_data_3841)    
    peak = wt    
    maxid = signal.argrelmax(data_3841_numpy, order=peak)
    print("ピーク位置:",maxid[0])
    print("maxid: ",maxid)    
    
    # 検出ボタンのピーク検出結果を描画する(before: この時点でのピークは, 単純にピーク検出したものを全て残すようにしている)
    Fig = plt.figure(figsize=(6, 6))
    Map1 = Fig.add_subplot(111)    
    left = list(range(0,len(new_data_3841),1))
    Map1.plot(left, data_3841_numpy)        
    #Map1.scatter(left[maxid[0]], data_3841_numpy[maxid[0]])
    for i in range(len(maxid[0])):
        Map1.text(left[maxid[0][i]], data_3841_numpy[maxid[0][i]], 'PEAK!')
    new_dir_path_recursive = './auto_output/{}/{}/drink_peak_graph'.format(datetime.date.today(),FileName)
    os.makedirs(new_dir_path_recursive, exist_ok=True)
    plt.savefig("{}/peak_{}_before.png".format(new_dir_path_recursive,count))
    plt.close()        
    
    crd_similar = []
    similar_to_find_max = []  # ピークの時の類似度を格納する配列. この配列から最大値を求める. この処理は ピークの域値を決める時(最大類似度*0.8)に使用する
    for i in maxid[0]:
        similar_to_find_max.append(new_data_3841[i])
        a = [i+w_move, new_data_3841[i]]    # ピークをボタンの中心座標に切り替えるため, w_moveを付与している
        crd_similar.append(a)
    #print("crd_similar = ",crd_similar)
    #print(len(crd_similar))

    # 検出されたピークの中から域値(最大類似度*0.8)以上のものだけを取り出す. 
    # (中心座標(検出位置 + w_move), 類似度) の形にする
    print("before_crd_similar = {}".format(crd_similar))
    print("len(before_crd_similar) = {}".format(len(crd_similar)))
    new_crd_similar = []
    for i in crd_similar:
        if i[1] >= max(similar_to_find_max) * th:  # 検出したピークの最大類似度の80%以上のピークのみを抽出する
            new_crd_similar.append(i)
    
    #print("after new_crd_similar",new_crd_similar)
    #print(len(new_crd_similar))

    # 検出ボタンのピークを特定の域値に従って抜き出した後のグラフを保存する
    maxid = []
    for i in new_crd_similar:
        maxid.append(i[0]-w_move)
    maxid = np.array(maxid)
    
    Fig = plt.figure(figsize=(6, 6))
    Map1 = Fig.add_subplot(111)    
    left = list(range(0,len(new_data_3841),1))
    #print("len(left): ",len(left))
    #print("len(data_3841_numpy): ",len(data_3841_numpy))    
    Map1.plot(left, data_3841_numpy)
    
    for i in range(len(maxid)):
        Map1.text(left[maxid[i]], data_3841_numpy[maxid[i]], 'PEAK!')
    new_dir_path_recursive = './auto_output/{}/{}/drink_peak_graph'.format(datetime.date.today(),FileName)
    os.makedirs(new_dir_path_recursive, exist_ok=True)
    plt.savefig("{}/peak_{}_after.png".format(new_dir_path_recursive,count))
    plt.close()        
    
    return new_crd_similar


# ボトルの切り出しと保存を行う関数.
# 検出した各列のピークを元にボトル画像を切り出していく
def crop_and_save_bottle_fig(new_crd_similar,temp,FileName,h_move,func,count):  # 関数化2
    # ボトルの切り出しを行う ================================= 
    # 隣り合うボタン同士の距離を比較する
    # ピークとピークの距離を求め, その平均を求める

    # 検出したピーク(おそらくそれはボタンであるはず) の座標だけを取り出して新しい配列に保存する
    only_peak_crd = []
    for i in new_crd_similar:
        only_peak_crd.append(i[0])
    print("only_peak_crd = ",only_peak_crd)
    
    # 隣り合うピーク同士の差分を求めて, 新しい配列に保存する
    only_peak_crd = np.array(only_peak_crd)
    peak_diff = np.diff(only_peak_crd)
    print("peak_diff = ",peak_diff)

    # 上のnp.diff関数は差分を取る際にマイナス値も出てしまうので、それを防ぐために中身を絶対値に直し,新しい配列に格納する
    peak_diff_abs = []
    for i in peak_diff:
        peak_diff_abs.append(i)

    # ピーク間同士の平均値を求める
    peak_diff_abs = np.array(peak_diff_abs)
    print("peak_diff_abs = ",peak_diff_abs)
    ave_between_peaks = sum(peak_diff_abs)/len(peak_diff_abs)
    print("ave_between_peaks = ",ave_between_peaks)
    ave_np = np.average(peak_diff_abs)
    print("ave_np = ",ave_np)

    # ボタンのテンプレートの大きさから500mlボトルのサイズを推定する(この処理は後の切り出しの際に使用する)
    ht,wt = temp.shape
    print()
    print("スケール変換後のテンプレの高さ・幅",ht,wt)
    crop_btl_h = int(wt * (210/44))  # これは元の実スケールとの対比を使用して求めている
    crop_btl_w = int(wt * (67/44))  # 実スケールは11/1のノートに記載されている
    print("crop_btl_h = ",crop_btl_h)
    print("crop_btl_w = ",crop_btl_w)

    # ドリンク画像を切り出す、もし画面外になった場合は,画面の端っこまでを切り出すという処理にする
    order = 1 # 左から何番目のドリンクかをファイル名に保存する際に使用する
    for i in only_peak_crd:
        img_to_crop = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
        # ドリンク画像の切り出し, 保存
        # img[top : bottom, left : right]
        
        top = int((func(i)) - (h_move) - (crop_btl_h))     # func の関数の中身 今は応急処置: -0.006538 x + 896        
        if top < 0:      # クロップする際に画面の端を超えてしまった場合は端っこまでをクロップすることにする
            top = 0
        bottom = int((func(i)) - (h_move))
        left = i-int(ave_between_peaks/2)
        right = i+int(ave_between_peaks/2)                
        img_crop = img_to_crop[ top: bottom, left: right]
        
        new_dir_path_recursive = './auto_output/{}/{}/drink_crop_result'.format(datetime.date.today(),FileName)
        os.makedirs(new_dir_path_recursive, exist_ok=True)
        cv2.imwrite("{}/crop_s{}_{}_l{}.jpg".format(new_dir_path_recursive,count,i,order), img_crop)
        order += 1

        
# 各段のボタンを抽出する関数
# ボタンをいくつ抽出するかは手動設定で行なっている
def get_choice_xy_sml(rank_idx,res_crop,res_at,res_top,wt):
    # ボタンを選び,抽出する処理を書く
    choice_xy_crd_sml = []        
    for i in range(len(rank_idx)):        
        line,colum = int(rank_idx[i]/len(res_crop[0])), (rank_idx[i] % len(res_crop[0]))
        xy_crds_sml = ([colum,line+res_top], res_at[line+res_top][colum])
        if len(choice_xy_crd_sml) == 0: # 選び出したボタンの座標と類似度を格納する配列が空ならそのまま代入する
            choice_xy_crd_sml.append(xy_crds_sml)
        else: # 選んだボタンが1つ以上であるならば,
            compare = []
            # その時点で選んである全ての検出座標と比較して,x座標間の差を取得する(同じ位置にあるようなボタンを使用するのを避けるため)
            for j in choice_xy_crd_sml: 
                compare.append(abs(j[0][0]-xy_crds_sml[0][0]))                     
            compare = np.array(compare)            
            # x座標同士を比較して,離れているボタンを抜き出す
            if (np.count_nonzero(compare <= wt/2)) == 0 : # 差が wt/2 以下のものが1つもないなら
                choice_xy_crd_sml.append(xy_crds_sml)
                    
        # ボタンを域値以上抜き出したら,抽出終了とする
        if len(choice_xy_crd_sml) >= 5:
            break
    return choice_xy_crd_sml


def main(): #=====================================================================================    
    print("自販機画像のファイル名(数字のみ)を入力してください. 例: 4245")
    FileName = str(input())
    print("テンプレート画像のファイル名(数字のみ)を入力してください. 例: 1962")
    TempName = str(input())
    print("ありがとう. 解析を開始します.\n")

    start = time.time()
    
    # 入力画像とテンプレート画像を取得
    img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
    temp = cv2.imread("../../temp_img/IMG_{}.jpg".format(TempName))
    
    gray_input = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)   
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

    # 入力画像の高さ・幅
    h_input,w_input = gray_input.shape
    print(h_input,w_input)

    # テンプレート画像の高さ・幅
    h, w = temp.shape
    print("元のテンプレの高さ,幅:",h,w)

    # 検出個数が最も多いときのスケールを取得する
    max_btn_nums_scl,temp = get_max_button_scale(gray_input,temp) 
    print("最も検出されたときの個数,スケール",max_btn_nums_scl)    
    ht,wt = temp.shape
    print("スケール変換後のテンプレの高さ・幅",ht,wt)
            
    # 検出個数が最も大きい時の スケール でもう一度テンプレートマッチング(OpenCV)
    # ここのテンプレマッチング結果を使用して、後々のボタン列のところをもう一度マッチングする処理を減らすことはできないだろうか
    res = cv2.matchTemplate(gray_input,temp,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_input, pt, (pt[0] + wt, pt[1] + ht), (0,0,255), 2)
    print("検出個数: {}\n".format(len(loc[0])))
    
    # rgb画像に域値以上の結果を描画する
    new_dir_path_recursive = './auto_output/{}/{}/btn_detect'.format(datetime.date.today(),FileName)
    os.makedirs(new_dir_path_recursive, exist_ok=True)
    cv2.imwrite("{}/ikiti_ijou_temp_{}_{}_{}.jpg".format(new_dir_path_recursive,max_btn_nums_scl[1],FileName,threshold), img_input)
    

    # 縦軸(検出個数), 横軸(画像の縦サイズ) の類似度グラフを作成する ========================
    left = list(range(0,h_input,1))
    tate_graph = [0]*h_input                
    # 左上のy座標ではなく,ボタンの中心座標を格納するよう変更
    w_move = int(wt/2)
    h_move = int(ht/2)
    for i in loc[0]:     
        tate_graph[i+h_move] += 1
    # グラフを保存する
    #plt.bar(left, tate_graph)    
    #plt.savefig("./output/3841_jpg/20191212/num_button_graph/button_num_detect_add_hmove.png")

    # 上記の縦軸(検出個数),横軸(画像の縦サイズ)のグラフのピークを検出する
    data_3841_numpy = np.array(tate_graph)
    maxid = signal.argrelmax(data_3841_numpy, order=int(wt/2))
    print("ピーク位置:",maxid[0])    

    # もし, ピークが2つしかない場合,域値を下げてピーク検出するようにここに書いても良い
    # resから域値が0.6以上を取り出し, それを使用してピークを求めるなど
    # 描画
    Fig = plt.figure(figsize=(6, 6))
    Map1 = Fig.add_subplot(111)
    Map1.plot(left, data_3841_numpy)    
    for i in range(len(maxid[0])):
        Map1.text(left[maxid[0][i]], data_3841_numpy[maxid[0][i]], 'PEAK!')
    # 保存
    new_dir_path_recursive = './auto_output/{}/{}/button_line_peak_graph'.format(datetime.date.today(),FileName)
    os.makedirs(new_dir_path_recursive, exist_ok=True)
    plt.savefig("{}/peak_before.png".format(new_dir_path_recursive))

    

    
    # ピーク位置を基準に上中下のボタンを取得する処理を加える
    # 各ボタン列の最大類似度を求める
    print("len(res): ",len(res))
    count = 1
    tate_graph_update = [0]*h_input # グラフ作成時に使用(検出個数を格納する配列)
    left_update = list(range(0,h_input,1)) # グラフ作成時に使用(グラフのx軸となる)
    res_alls = []
    for i in maxid[0]: # ピーク位置: [1061 1833 2603]
        print("ピーク位置:{}".format(i))
        # res を分割する
        res_top,res_bottom = i-(2*h_move),i+(2*h_move)     
        res_first = res[ res_top : res_bottom+1 ]      # 1回目のテンプレマッチングの結果を使用している
        # 分割した中での最大類似度を持つボタンを求める
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_first)
        print("分割した1段目のmax_val: {}".format(max_val))        
        x,y = max_loc[0],max_loc[1]+res_top  
        print("x,y = ",x,y)
        print("オリジナルresの同座標の類似度: {}\n".format( res[y][x] ))
        # 各段からボタンを切り出す(img[top : bottom, left : right])
        top,bottom,left,right = y,y+ht,x,x+wt
        img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
        btn_img = img_input[ top:bottom,left:right ]        
        # 切り出したボタン画像を保存する(確認のため)
        new_dir_path_recursive = './auto_output/{}/{}/each_section_temps'.format(datetime.date.today(),FileName)
        os.makedirs(new_dir_path_recursive, exist_ok=True)
        cv2.imwrite("{}/btn_{}.jpg".format(new_dir_path_recursive,count), btn_img)
        # 切り出したボタン画像を使用してテンプレマッチングを行う
        btn_img = cv2.cvtColor(btn_img, cv2.COLOR_RGB2GRAY)   
        res_update = cv2.matchTemplate(gray_input, btn_img, cv2.TM_CCOEFF_NORMED)
        res_alls.append(res_update) # それぞれの段で検出したマッチング結果を後々使用するため, ここに入れている
        threshold = 0.7
        loc_update = np.where( res_update >= threshold)        
        # 新しいテンプレを使用した時の各段の最大類似度
        res_second = res_update[ res_top : res_bottom+1 ]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_second)
        print("新しいテンプレを使用して分割した1段目のmax_val(): {}".format(max_val))        
        # 縦軸(検出個数),横軸(画像の縦サイズ)のグラフを描画する
        for i in loc_update[0]:     # ここで検出個数をカウントしている
            tate_graph_update[i+h_move] += 1 # h_move を加えることで, 検出ボタンの中心を保存している
        print("sum(tate_graph_update): {}\n".format(sum(tate_graph_update)))
        # 新しいテンプレートを使用して,マッチングを行なった際の検出結果を保存する
        # rgb画像に域値以上の結果を描画する
        img_input = cv2.imread("../../vm_full_img/IMG_{}.jpg".format(FileName))
        for pt in zip(*loc_update[::-1]):
            cv2.rectangle(img_input, pt, (pt[0] + wt, pt[1] + ht), (0,0,255), 2)        
        new_dir_path_recursive = './auto_output/{}/{}/new_temp_detect'.format(datetime.date.today(),FileName)
        os.makedirs(new_dir_path_recursive, exist_ok=True)
        cv2.imwrite("{}/ikiti_ijou_temp{}_{}_{}.jpg".format(new_dir_path_recursive,count,FileName,threshold), img_input)        
        count += 1
        
    # ピークが検出されなかった時は,再度テンプレマッチングの結果を使用して, ピークを見るのも良い.
    
    
    # 各段で検出したボタンの上位10%を使用して,その中心を通るような回帰直線を作成する
    count = 1
    drink_num = []
    ps_num = len(maxid[0])
    for i in range(len(maxid[0])): # [1061 1833 2603]
        print()
        # 1段ずつ処理を行っていく
        res_at = res_alls[i]
        peak_at = maxid[0][i]
        res_top,res_bottom = peak_at-(2*h_move),peak_at+(2*h_move)
        # 各段の特定範囲における検出個数を求める        
        res_crop = res_at[ res_top : res_bottom+1 ]        
        loc_each_section = np.where( res_crop >= threshold)
        detect_num = len(loc_each_section[0])
        # res_crop 配列を平坦化する
        res_crop_flat = np.ravel(res_crop)
        # 類似度が大きい順のインデックスを取得する
        rank_idx = res_crop_flat.argsort()[::-1]        
        # 平坦化された配列から,2次元配列時の座標を求めるために,行(line)と列(colum)の2つのインデックスが必要        
        line,colum = int(rank_idx[0]/len(res_crop[0])), (rank_idx[0] % len(res_crop[0]))                
        # res_at と res_cropで結果が同じであるかを確認する
        print("res_crop_flat[rank_idx[0]] = ",res_crop_flat[rank_idx[0]])
        print("res_at[line+res_top][colum] = ",res_at[line+res_top][colum])
        print("res_crop[line][colum] = ",res_crop[line][colum])
        print()
        # ボタンを選び出す
        choice_xy_sml = get_choice_xy_sml(rank_idx,res_crop,res_at,res_top,wt)
        print("{}回目:choice_xy_sml = {}\n".format(count,choice_xy_sml))
            
        # 回帰直線を引く
        x_points,y_points = get_xy_points(choice_xy_sml, w_move, h_move) # 検出したボタンの中心のx座標,y座標を取得        
        print("1次関数m,n = {}\n".format(np.polyfit(x_points, y_points, 1))) # 得られた中心のx,y座標から, 回帰直線を求める (numpy 配列でないと行けない)
        func = np.poly1d(np.polyfit(x_points, y_points, 1))  # 実際に回帰直線の関数を取得    
        new_line_crd = get_new_line_crd(func,w_input,w_move,h_move,wt)  # テンプレマッチング用の中心座標をこの配列で定義(new_line_crdには直線の座標が格納されているはず)        
        # 回帰直線上の類似度を求める
        ruijido_list = []
        for j in new_line_crd:            
            x_crd, y_crd = j[0],j[1]-h_move            
            ruijido_list.append(res_at[y_crd][x_crd])  # res_atに変えないといけない？
        print("len(ruijido_list) = {}\n".format(len(ruijido_list)))
        th = 0.5 # 類似度を抽出するときに、(最大類似度*th)の形で指定している。その際にこれを使用する
        # ピークを検出する
        new_crd_similar = get_peak_of_ruijido_list(ruijido_list,th,wt,w_move,count,FileName)
        print("new_crd_similar = ",new_crd_similar)
        print("len(new_crd_similar) = ",len(new_crd_similar))
        # 検出したボタンの数(=ドリンク数) を保存する
        drink_num.append(len(new_crd_similar))        
        # ボトルの切り出しと保存を行う        
        crop_and_save_bottle_fig(new_crd_similar,temp,FileName,h_move,func,count) # この関数は2つの処理を1つの関数内で行なっている。もっと分割できるはず
        count += 1                    

    print("\n\n")
    print("{}段 x {}本の自販機です".format(ps_num,max(drink_num)))    
    
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    
    exit()
        
#=====================================#    




if __name__ == "__main__":
    main()
