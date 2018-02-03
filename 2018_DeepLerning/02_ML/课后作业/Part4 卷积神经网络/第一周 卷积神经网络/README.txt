如果使用的tensorflow版本1.4的话，可能不会得到期望的输出结果，可能会出现类似下面这个帖子中的问题
https://mooc.study.163.com/learn/2001281004?tid=2001392030&forcelogin=true&edusave=1#/learn/forumdetail?pid=2001702006

经检查，发现使用一模一样的代码去跑的时候，还是失败，所以我们思考可能是tensorflow版本的问题，所以切换到tensorflow1.2

为了不破坏之前tensorflow1.4的环境，我们单独配置一个tensorflow1.2环境来跑我们的程序

步骤如下：
1.安装virtualenv 可参考 https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001432712108300322c61f256c74803b43bfd65c6f8d0d0000#0
2.安装项目需要的包 scipy、numpy、h5py、matplotlib、tensorflow1.2、jupyter，安装过程不在追溯，使用whl文件安装成功率高。



修正后代码可参考： https://github.com/RookieDay/py_example/tree/master/tf_1.2_deeplerning/tf12fix_deepai

