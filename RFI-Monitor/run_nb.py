from DataGen import *
#from Helper import *
from Config import Config
from Net import Network
from Plotting import *
from DataHandle import *
import h5py
config = Config()

#train_data, test_data, train_labels, test_labels = gen_TestTrain();
#plot_25_ims()
#plt.show()

net = Network()
load = False

if load:
    net.load()
else:
    train_data, test_data, train_labels, test_labels = gen_TestTrain();
    net.train(train_data, train_labels, 100)
    net.save()

datafile = 'Data/1460.npy'

exit(0)

#try:
#    data = np.load(datafile)
#    data = np.clip(data,0,5)/5
#    x = 2245
#    y = 3250
#    c = data[y:3600, x: x+32]
#except:
#    data = gen_VideoTest()     
x = 0
y = 0
#    c = data[:,:]

c = np.transpose(h5py.File('../SpecTools/test.h5')['psd'][32718:32718+32, :1200])
c = 10.*np.log10(c)
c = np.clip(c, 0, 5)/5.
print (c.shape)



#%matplotlib notebook

fig, ax = plt.subplots(1,1)
plt.ion()
fig.show()

buffer = ImgBuffer()
for n in range(np.shape(c)[0]-config.L):
    plt.cla()
    currIm = np.reshape(c[n:n+config.L,:], [1, config.L, config.L])
    pred = net.predict(currIm) 

    boxes = process_pred(pred)

    print (boxes)
    buffer.update_timer()
    buffer.process_new(boxes, (y+n), x)
    buffer.process_existing()
    
    plt.cla()
    ax.imshow(c[n:n+config.L,:], vmax = 1, vmin = 0)
    plt.xticks(np.arange(config.f, config.L, config.f))
    plt.yticks(np.arange(config.f, config.L, config.f))
    plt.axis("on")
    plt.title(n)
    plt.grid(True, alpha = .3)
    for z in range(len(boxes)):
        cx, cy, w, h = boxes[z]
        ax.add_patch(patch.Circle((cy,cx), 0.5, ec = 'w', fc = 'w'))
        ax.add_patch(patch.Rectangle((cy-h/2, cx-w/2),\
            h, w, ec='w', fc='none'))  
    
    fig.canvas.draw() 
buffer.clear_buffer()    