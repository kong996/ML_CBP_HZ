import rebound
import numpy as np
from multiprocessing import Process, Lock
def setupSimulation(u, e, a_p, inc):
    #Initial Condition
    m1 = 1-u
    m2 = u
    m3 = 3e-6    #earth mass
    #Set up the model
    sim = rebound.Simulation()
    sim.add(m=m1) #Star A
    sim.add(m=m2, a=0.1, e=e) #Star B
    sim.add(m=m3, a=a_p, inc=inc) #Planet ABc
    sim.move_to_com()
    return sim
#Initial Condition for P-type with inclination and phase
total_size = 400000
process_num = 20
each_size = int(total_size/process_num)
each_num = np.linspace(0, each_size, each_size, endpoint=False, dtype=int)
#The initial condition of simulation
#t = 2*np.pi
u = np.random.randint(1, 51, size=total_size)/100
e = np.random.randint(0, 100, size=total_size)/100
a_p = np.random.randint(150, 601, size=total_size)/100
inc = np.random.randint(0, 19, size=total_size)*np.pi/18

def samples(process_id, job_id):
    start_data = (job_id-1)*len(each_num)
    print('Process Id {0} Job Id {1}'.format(process_id, job_id))
    for i, j in enumerate(each_num):
        uu = u[j+start_data]
        ee = e[j+start_data]
        a_pp = a_p[j+start_data]
        incc = inc[j+start_data]
        
        v_esc2 = 2/a_pp # the square of v_escape
        sim = setupSimulation(uu, ee, a_pp, incc)
        #sim.exit_min_distance = 0.01 #hill radius
        Noutputs = 1000*4
        t = 2*np.pi*a_pp*np.sqrt(a_pp)
        times = np.linspace(0, 1000*t, Noutputs)
        distances = np.zeros(Noutputs)
        velocity2 = np.zeros(Noutputs)
        ps = sim.particles # ps is now an array of pointers. It will update as the simulation ru
        result = 0
        #try:
        for i, time in enumerate(times):
            sim.integrate(time)
            dp = ps[1] - ps[2]
            dv = ps[2]
            distances[i] = np.sqrt(dp.x*dp.x+dp.y*dp.y+dp.z*dp.z)
            velocity2[i] =  (dv.vx*dv.vx+dv.vy*dv.vy+dv.vz*dv.vz)
        #except rebound.Encounter as error:
            #result = -1
        #else:
        if max(velocity2) > v_esc2 or min(distances) < 1.:
            result = 0
        else:
            result = 1
        with open('earth_train{0}.csv'.format(job_id), 'a') as f:
            f.write(str(u[j+start_data]))
            f.write(' ')
            f.write(str(e[j+start_data]))
            f.write(' ')
            f.write(str(a_p[j+start_data]))
            f.write(' ')
            f.write(str(inc[j+start_data]))
            f.write(' ')
            f.write(str(result))
            f.write('\n')
process_dict = {}
for i in range(process_num):
    process_dict.update({'Process-%d'%(i+1):i+1})
for process_id, job_id in process_dict.items():
    p=Process(target=samples, args=(process_id, job_id))
    p.start()
print('exiting Main Process')