import numpy as np
import time
import matplotlib.pylab as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

pix=4.289e-3
twopi=2*np.pi
au=1.496e8 # km
RJup=7.1492e4 # km

fh=open("pos.csv","r")
lines = fh.readlines()

tstamp=[]
dist=[]
fl=[]
sig=[]
geod=[]
logdata=[]

lognum=0

for line in lines[1:]:
    vals=line.rstrip().split(",")
    t = time.strptime(vals[0]+" "+vals[1],"%d-%b-%y %I:%M:%S %p")
    tstamp.append(t[0]*1000+t[7]+(t[3]+t[4]/60+t[5]/3600)/24)
    try: a=float(vals[7])
    except: a=vals[7]
    try: b=float(vals[10])
    except: b=vals[10]
    try: c=float(vals[13])
    except: c=vals[13]
    try: d=float(vals[16])
    except: d=vals[16]
    dist.append([a,b,c,d])
    fl.append(float(vals[17]))
    sig.append(float(vals[18]))
    geod.append(float(vals[19]))

sig=np.array(sig)
fl=np.array(fl)
dist=np.array(dist)
geod=np.array(geod)
tstam=np.array(tstamp)

sig*=pix/fl*206265.

tplt=[]
dplt=[]
splt=[]

for i in range(len(tstamp)):
    convert=pix/fl[i]*geod[i]*au/RJup
    print(convert,pix,fl[i],geod[i])
    for d in range(4):
        if dist[i,d]=="NA": continue
        tplt.append(tstamp[i])
        dplt.append(float(dist[i,d])*convert)
        splt.append(sig[i])

tplt=np.array(tplt)
dplt=np.array(dplt)
splt=np.array(splt)


# Now let's see about finding periods and phases.


P_init=5
P_min=0.1
P_max=25
delta_P=0.05
F_init=np.pi
F_min=0
F_max=twopi
A_init=10
A_min=1
A_max=50


fig=plt.figure(figsize=(8,5),dpi=100)

curv_ax=plt.axes([0.28,0.33,0.65,0.65])
slider_P_ax=plt.axes([0.1,0.01,0.65,0.025])
slider_A_ax=plt.axes([0.1,0.06,0.65,0.025])
slider_F_ax=plt.axes([0.1,0.11,0.65,0.025])
input_P_ax=plt.axes([0.10,0.17,0.075,0.05])

radio_ax=plt.axes([0.025,0.83,0.15,0.15])
radio = RadioButtons(radio_ax,('None-Grey','M1-Red','M2-Green','M3-Blue','M4-Purple'),active=0)

def colorfunc(label):
    print("Selected {}".format(label))

radio.on_clicked(colorfunc)

scat=curv_ax.scatter(tplt,dplt,c=[[0.5,0.5,0.5,1]]*len(tplt),picker=True)
curv_ax.set_ylim(-30,30)
curv_ax.set_xlabel("Day")
curv_ax.set_ylabel('Jupiter Radii')

def on_pick(event):
    colour_label=radio.value_selected
    #print("Colour label is {}".format(colour_label))
    if colour_label=='None-Grey':colour=[[0.5,0.5,0.5,1]]
    elif colour_label=='M1-Red':colour=[[1,0,0,1]]
    elif colour_label=='M2-Green':colour=[[0,1,0,1]]
    elif colour_label=='M3-Blue':colour=[[0,0,1,1]]
    elif colour_label=='M4-Purple':colour=[[1,0,1,1]]
    else: colour=[[0.5,0.5,0.5,1]]
    #print(colour)
    #print("clicked {}".format(tplt[event.ind]))
    #scat._facecolors[event.ind,:] = colour
    #scat._edgecolors[event.ind,:] = colour
    c_edge=scat.get_edgecolors()
    c_face=scat.get_facecolors()
    c_edge[event.ind]=colour
    c_face[event.ind]=colour
    scat.set_edgecolor(c_edge)
    scat.set_facecolor(c_face)
    curv_ax.draw_artist(scat)
    #fig.canvas.draw()


t=np.linspace(tplt[0],tplt[-1],1000)


plt.axes(curv_ax)
cos_plot, = plt.plot(t,A_init*np.cos( (t-t[0])*twopi/P_init+F_init))


P_slider = Slider(slider_P_ax,'P (d)',P_min,P_max,valinit=P_init, valstep=delta_P)
A_slider = Slider(slider_A_ax,'A (RJ)',A_min,A_max,valinit=A_init)
F_slider = Slider(slider_F_ax,'F (rad)',F_min,F_max,valinit=F_init)
P_input = TextBox(input_P_ax,'Set P ')

def update(val):
    P = P_slider.val
    A = A_slider.val
    F = F_slider.val
    cos_plot.set_ydata(A*np.cos( (t-t[0])*twopi/P+F))
    curv_ax.draw_artist(cos_plot)
    #fig.canvas.blit(fig.bbox)
    #fig.canvas.draw()

def textSubSlide(val):
     P_slider.set_val(float(val))

P_slider.on_changed(update)
A_slider.on_changed(update)
F_slider.on_changed(update)

P_input.on_submit(textSubSlide)


log_ax = plt.axes([0.025,0.68,0.15,0.05])
button_log=Button(log_ax,'Log Values',color='lightgreen',hovercolor='0.975')

def log(event):
    # Print the current log values to the plot window.
    global lognum, logdata
    max_index = len(logdata)-1
    slider_vals = [F_slider.val, A_slider.val, P_slider.val]
    if len(logdata) > 0 and logdata[max_index] == slider_vals:
        return
    else:
        for txt in fig.texts:
            txt.set_visible(False)
        logdata.append(slider_vals)
        lognum = (len(logdata)-1)
        fig.text(0.95,-0.395,'{0:.3f} \n{1:.3f} \n{2:.3f}'.format(slider_vals[0],
                                                                 slider_vals[1],
                                                                 slider_vals[2]),
                                                          horizontalalignment='center',
                                                          verticalalignment='center',
                                                          transform=curv_ax.transAxes,
                                                          linespacing=2,
                                                          weight='bold')
        fig.text(0.95,-0.23,'Logged Potential\nSolution {}:'.format(lognum+1),
                                                             horizontalalignment='center',
                                                             verticalalignment='center',
                                                             transform=curv_ax.transAxes,
                                                             fontsize=8)
        # Print the current log values to the terminal.
        print("\nLogged Potential Solution:\
              \nPeriod: {} d\
              \nAmplitude: {} Jupiter Radii\
              \nPhase: {} rad".format(P_slider.val,A_slider.val,F_slider.val))
        fig.canvas.draw()

button_log.on_clicked(log)


# A button to clear the log separately of the plot.
clear_log_ax = plt.axes([0.025,0.48,0.15,0.05])
button_clear_log=Button(clear_log_ax,'Clear Log',color="lightblue",hovercolor='0.975')

def clear_log(event):
    global lognum, logdata
    for txt in fig.texts:
        txt.set_visible(False)
    logdata=[]
    lognum=0
    fig.canvas.draw_idle() # Updates the canvas immediately.

button_clear_log.on_clicked(clear_log)


# Buttons to move forward and backward through the logged data.
prev_log_ax = plt.axes([0.025,0.58,0.057,0.05])
button_prev_log=Button(prev_log_ax,'◄',color="lightpink",hovercolor='0.975')

def prev_log(event):
    global lognum, logdata
    if lognum == 0:
        return
    else:
        lognum -= 1
        p_data = logdata[lognum]
        for txt in fig.texts:
            txt.set_visible(False)
        fig.text(0.95,-0.395,'{0:.3f} \n{1:.3f} \n{2:.3f}'.format(p_data[0],
                                                                 p_data[1],
                                                                 p_data[2]),
                                                         horizontalalignment='center',
                                                         verticalalignment='center',
                                                         transform=curv_ax.transAxes,
                                                         linespacing=2,
                                                         weight='bold')
        fig.text(0.95,-0.23,'Logged Potential\nSolution {}:'.format(lognum+1),
                                                             horizontalalignment='center',
                                                             verticalalignment='center',
                                                             transform=curv_ax.transAxes,
                                                             fontsize=8)
        fig.canvas.draw_idle()

button_prev_log.on_clicked(prev_log)


next_log_ax = plt.axes([0.12,0.58,0.057,0.05])
button_next_log=Button(next_log_ax,'►',color="lightpink",hovercolor='0.975')

def next_log(event):
    global lognum, logdata
    if lognum+1 > (len(logdata)-1):
        return
    else:
        lognum += 1
        n_data = logdata[lognum]
        for txt in fig.texts:
            txt.set_visible(False)
        fig.text(0.95,-0.395,'{0:.3f} \n{1:.3f} \n{2:.3f}'.format(n_data[0],
                                                                 n_data[1],
                                                                 n_data[2]),
                                                         horizontalalignment='center',
                                                         verticalalignment='center',
                                                         transform=curv_ax.transAxes,
                                                         linespacing=2,
                                                         weight='bold')
        fig.text(0.95,-0.23,'Logged Potential\nSolution {}:'.format(lognum+1),
                                                             horizontalalignment='center',
                                                             verticalalignment='center',
                                                             transform=curv_ax.transAxes,
                                                             fontsize=8)
        fig.canvas.draw_idle()

button_next_log.on_clicked(next_log)


reset_ax = plt.axes([0.025, 0.33, 0.15, 0.05])
button = Button(reset_ax, 'Reset Plot', color="yellow", hovercolor='0.975')

def reset(event):
    P_slider.reset()
    A_slider.reset()
    F_slider.reset()
    c=scat.get_facecolors()
    c[:]=[[0.5,0.5,0.5,1]]
    scat.set_edgecolor(c)
    scat.set_facecolor(c)
    fig.canvas.draw()
button.on_clicked(reset)

#connection_id = fig.canvas.mpl_connect('button_press_event', on_pick)
cid=fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
