#验证码识别

使用resnet网络作为backbone。因验证码长度固定，识别字符有0-9和A-Z共36个字符，
因此可直接将resnet网络输出为验证码长度乘于字符数（即5*36=180）。最后将输出
reshape为[B, 5, 36]的形式作为最终的输出。

##安装
`pip install -r requirements.txt`

##识别
`python inference.py --data_path ''`

`data_path: 检测图片所在目录路径或者需要检测的单张图片路径`

##说明
检测结果保存在result.txt中，也可以指定保存的路径`--output_dir`
