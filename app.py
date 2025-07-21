# 在导入matplotlib之前设置后端
import matplotlib
matplotlib.use('Agg')  # 添加这一行
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import io
import base64
import matplotlib as mpl
from datetime import datetime
import socket
from werkzeug.serving import run_simple

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
mpl.rcParams['font.size'] = 12

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# 确保图片目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 树种参数定义
species_params = {
    '华山松': [24.994, 42.728, 1.032, 22.717, 0.182, 24.893, 0.182, 16.132, 0.233, 6.572, 1.475],
    '油松': [33.445, 39.62, 0.878, 19.872, 0.221, 25.564, 0.466, 23.407, 0.121, 6.631, 1.584],
    '锐齿栎': [16.713, 33.527, 1.189, 40.094, 0.133, 39.972, 0.058, 12.812, 0.300, 4.951, 2.086]
}

def calculate_stand_parameters(species, sci, sdi, ages):
    """
    计算林分参数
    :param species: 树种名称
    :param sci: 地位级指数
    :param sdi: 林分密度指数
    :param ages: 年龄列表
    :return: 包含所有林分参数的DataFrame
    """
    params = species_params[species]
    H_a, H_b, H_c, D_a, D_b, D_c, D_d, G_a, G_b, G_c, G_d = params
    
    # 计算基准年龄时的树高（用于调整）
    if species == '锐齿栎':  # 锐齿栎的基准年龄为20年
        base_age = 20
        H0 = H_a / (1 + H_b * base_age ** (-H_c))
    else:
        base_age = 30
        H0 = H_a / (1 + H_b * base_age ** (-H_c))
    
    results = []
    
    for i, age in enumerate(ages):
        # 1. 计算平均树高 (m)
        H = H_a / (1 + H_b * age ** (-H_c)) * (sci / H0)
        
        # 2. 计算平均胸径 (cm)
        D = D_a * sci ** D_b * np.exp(-D_c * (sdi/1000) ** D_d / age)
        
        # 3. 计算林分断面积 (m²/ha)
        G = G_a * sci ** G_b * np.exp(-G_c * (sdi/1000) ** (-G_d) / age)
        
        # 4. 计算林分密度 (株/ha)
        N = G * 40000 / (np.pi * D**2)
        
        # 5. 计算林分蓄积量 (m³/ha) - 使用形高法: V = G * (H + 3) * f, f取0.43
        V = G * (H + 3) * 0.43
        
        # 6. 计算平均单株材积 (m³/株)
        v = V / N if N > 0 else 0
        
        # 7. 计算平均生长量 (m³/ha/yr)
        MAI = V / age
        
        # 8. 计算连年生长量 (m³/ha/5yr)
        if i == 0:
            CAI = 0
        else:
            prev_V = results[i-1][5]  # 前一年的蓄积量
            CAI = (V - prev_V) / (ages[i] - ages[i-1])
        
        # 9. 计算蓄积生长率 (%)
        if i == 0:
            P = 0
        else:
            prev_V = results[i-1][5]  # 前一年的蓄积量
            P = CAI * 200 / (prev_V + V)  # 公式: P = (CAI * 2) / (V_{t-1} + V_t) * 100%
        
        # 10. 计算密度指数 (Reineke指数)
        SDI_reineke = N * (D/20)**1.662
        
        results.append(np.round([
            age, H, D, G, N, v, V, MAI, CAI, P, SDI_reineke
        ],2))
    
    columns = [
        '年龄(yr)', '平均树高(m)', '平均胸径(cm)', '林分断面积(m²/ha)', 
        '林分密度(株/ha)', '平均单株材积(m³/株)', '林分蓄积量(m³/ha)',
        '平均生长量(m³/ha/yr)', '连年生长量(m³/ha/5yr)', '蓄积生长率(%)', '密度指数'
    ]
    
    return pd.DataFrame(results, columns=columns)

def create_plots(df, species, sci, sdi):
    """创建图表并返回图像文件名"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img_files = []
    
    # 1. 林分蓄积量变化
    plt.figure(figsize=(10, 6))
    plt.plot(df['年龄(yr)'], df['林分蓄积量(m³/ha)'], 'o-', linewidth=2, markersize=6, label=species)
    plt.xlabel('年龄 (年)')
    plt.ylabel('林分蓄积量 (m3/ha)')
    plt.title(f'{species}林分蓄积量随年龄变化 (SCI={sci}, SDI={sdi})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    img1 = f'volume_{timestamp}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], img1))
    plt.close()
    img_files.append(img1)
    
    # 2. 平均树高变化
    plt.figure(figsize=(10, 6))
    plt.plot(df['年龄(yr)'], df['平均树高(m)'], 's-', linewidth=2, markersize=6, label=species)
    plt.xlabel('年龄 (年)')
    plt.ylabel('平均树高 (m)')
    plt.title(f'{species}平均树高随年龄变化 (SCI={sci}, SDI={sdi})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    img2 = f'height_{timestamp}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], img2))
    plt.close()
    img_files.append(img2)
    
    # 3. 林分断面积变化
    plt.figure(figsize=(10, 6))
    plt.plot(df['年龄(yr)'], df['林分断面积(m²/ha)'], 'd-', linewidth=2, markersize=6, label=species)
    plt.xlabel('年龄 (年)')
    plt.ylabel('林分断面积 (m2/ha)')
    plt.title(f'{species}林分断面积随年龄变化 (SCI={sci}, SDI={sdi})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    img3 = f'basal_area_{timestamp}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], img3))
    plt.close()
    img_files.append(img3)
    
    # 4. 林分密度变化
    plt.figure(figsize=(10, 6))
    plt.plot(df['年龄(yr)'], df['林分密度(株/ha)'], '^-', linewidth=2, markersize=6, label=species)
    plt.xlabel('年龄 (年)')
    plt.ylabel('林分密度 (株/ha)')
    plt.title(f'{species}林分密度随年龄变化 (SCI={sci}, SDI={sdi})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    img4 = f'density_{timestamp}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], img4))
    plt.close()
    img_files.append(img4)
    
    return img_files

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', species_list=list(species_params.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    # 获取表单数据
    species = request.form['species']
    sci = float(request.form['sci'])
    sdi = float(request.form['sdi'])
    
    # 年龄范围
    ages = list(range(20, 81, 5))
    
    # 计算林分参数
    df = calculate_stand_parameters(species, sci, sdi, ages)
    
    # 创建图表
    img_files = create_plots(df, species, sci, sdi)
    
    # 将DataFrame转换为HTML表格
    table_html = df.to_html(
        classes='table table-striped table-bordered text-center align-middle',
        index=False,
        justify='center'
    )

    from datetime import datetime  # 确保顶部已导入
    now = datetime.now()
    
    # 渲染结果页面
    return render_template(
        'results.html',
        species=species,
        sci=sci,
        sdi=sdi,
        table_html=table_html,
        img_files=img_files,
        now=now
    )

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template

# 替换现有的运行代码
# 替换最后的运行代码为：
if __name__ == '__main__':
    # 获取本地IP地址
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"服务器将在 http://{local_ip}:5000 运行")
    print("在同一网络中的其他设备可以通过此地址访问")

    # 运行应用
    run_simple(hostname='0.0.0.0', port=5000, application=app, threaded=True)