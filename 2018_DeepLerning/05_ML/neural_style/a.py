# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
import numpy as np  
import matplotlib.pyplot as plt  

models ={'steps':steps,'used_times':used_times,'all_loss':{'content_loss':c_loss,'style_loss':s_loss,'tv_loss':t_loss,'loss':los}} 
models ={'steps':[1,2,3],'used_times':'11','all_loss':{'content_loss':[1,2,3],'style_loss':[1,2,3],'tv_loss':[1,2,3],'loss':[1,2,3]}} 

plt.figure(figsize=(14,8))
i = 0
print(models['all_loss'])
for loss_key,loss_val in models['all_loss'].items():
    print(loss_key,loss_val)
    plt.subplot(221+i) 
    plt.tight_layout()
    plt.plot(models['steps'],models['all_loss'][loss_key],label=str(loss_key))
    plt.ylabel(loss_key) 
    plt.xlabel('per steps')
    plt.title(loss_key + models['used_times'])
    legend = plt.legend(loc='best',shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    i = i + 1
plt.savefig("loss_chart" +".jpg")
plt.show()
