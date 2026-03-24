from scipy import integrate # for numerical integration
import csv # for importing data from csv files
import numpy as np # for general numerical operations
import matplotlib.pyplot as plt # for making plots
import scipy.odr as odr # for curve fitting with multivariate uncertainty
import scipy.stats as stats # for statistical calculations

###############################################################################################################
# define the fit function
# B is an array of the fit parameters, and x is the independent variable
def f(B, x): 
    return np.piecewise(x, 
                        [
                            x < B[0], 
                            (B[0] <= x) & (x <= B[1])
                        ], 
                        [
                            lambda x: B[2], 
                            lambda x: B[2] - B[3]*(x-B[0]), 
                            lambda x: (B[2] - B[3]*(B[1]-B[0]))*np.exp(-(x-B[1])/B[4])
                        ])

 # a model for the field decay

###############################################################################################################
# data for the 17 Ohm shunt resistor
times = []
voltages1 = []

with open('ALL0001/F0001CH2.csv', 'r', newline='') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    next(csvReader, None)
    for row in csvReader:
        times.append(float(row[3]))
        voltages1.append(float(row[4]))

offset1 = np.mean(voltages1[1:20]) # find the initial value during the time before the field switches

voltages1adj = [val - offset1 for val in voltages1] # subtract the initial value

fields1 = integrate.cumulative_trapezoid(voltages1adj) # numerically integrate dB/dt
min1 = np.min(fields1)
max1 = np.max(fields1)
fields1adj = [(val - min1) / (max1 - min1) for val in fields1] # normalize the signal 

p01 = [0.0000036, 0.0000712, 1, 14000, .00007] # initial guess for fit parameters: offset, slope

x_guess = np.linspace(np.min(times[1:]), np.max(times[1:]), 1000) # this defines a range from the minimum of our data to the max of our data, and splits it up into 100 evenly-spaced values
y_guess1 = f(p01, x_guess) # this applies our fit function to the entire range defined above

# set up and perform the non-linear regression
myfunc = odr.Model(f) # put our fit function into a special container
mydata1 = odr.RealData(times[1:], fields1adj) # put all our data into a special container
myodr1 = odr.ODR(mydata1, myfunc, beta0 = p01, sstol = 1e-20, job=00000) # set up a fitting data structure
myoutput1 = myodr1.run() # perform the nonlinear regression to find best-fit parameters

sd1  = myoutput1.sd_beta # the normalized standard deviations
p1   = myoutput1.beta # the best-fit parameters

print(p1) # if the fit converged, we should see a list of two parameters: the offset and the slope
print(sd1) # this gives the SD for the model parameters

x_fit = np.linspace(np.min(times[1:]), np.max(times[1:]), 1000) # this defines a range from the minimum of our data to the max of our data, and splits it up into 100 evenly-spaced values
y_fit1 = f(p1, x_fit) # this applies our fit function to the entire range defined above

plt.plot(times[1:], fields1adj,label='Data')
plt.plot(x_guess, y_guess1,'g',label='Guess')
plt.plot(x_fit, y_fit1,'r',label='Best Fit')
plt.xlabel('Time since switch (s)')
plt.ylabel('Normalized magnetic field (A.U.)')
plt.legend(loc=1)
plt.title(r'$R_s=17~\Omega$')
plt.show()


###################################################################################################################
# data for the 1.0 Ohm shunt resistor

voltages2 = []

with open('ALL0002/F0002CH2.csv', 'r', newline='') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    next(csvReader, None)
    for row in csvReader:
        voltages2.append(float(row[4]))


offset2 = np.mean(voltages2[1:20])
voltages2adj = [val - offset2 for val in voltages2]

fields2 = integrate.cumulative_trapezoid(voltages2adj)
fields2adj = [(val -min1)/(max1-min1) for val in fields2]

p02 = [0.000004, 0.00006, 1, 13500, .00004] # initial guess for fit parameters: offset, slope

y_guess2 = f(p02, x_guess) # this applies our fit function to the entire range defined above

# set up and perform the non-linear regression
mydata2 = odr.RealData(times[1:], fields2adj) # put all our data into a special container
myodr2 = odr.ODR(mydata2, myfunc, beta0 = p02, sstol = 1e-20, job=00000) # set up a fitting data structure
myoutput2 = myodr2.run() # perform the nonlinear regression to find best-fit parameters

sd2  = myoutput2.sd_beta # the normalized standard deviations
p2   = myoutput2.beta # the best-fit parameters

print(p2) # if the fit converged, we should see a list of two parameters: the offset and the slope
print(sd2) # this gives the SD for the model parameters

y_fit2 = f(p2, x_fit) # this applies our fit function to the entire range defined above

plt.plot(times[1:], fields2adj,label='Data')
plt.plot(x_guess, y_guess2,'g',label='Guess')
plt.plot(x_fit, y_fit2,'r',label='Best Fit')
plt.xlabel('Time since switch (s)')
plt.ylabel('Normalized magnetic field (A.U.)')
plt.title(r'$R_s=1.0~\Omega$')
plt.legend(loc=1)
plt.show()

###################################################################################################################
# data for the 0.1 Ohm shunt resistor

voltages3 = []

with open('ALL0003/F0003CH2.csv', 'r', newline='') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    next(csvReader, None)
    for row in csvReader:
        voltages3.append(float(row[4]))

offset3 = np.mean(voltages3[1:20])
voltages3adj = [val - offset3 for val in voltages3]

fields3 = integrate.cumulative_trapezoid(voltages3adj)
fields3adj = [(val -min1)/(max1-min1) for val in fields3]

p03 = [0.0000035, 0.000025, 1, 9500, .0002] # initial guess for fit parameters: offset, slope

y_guess3 = f(p03, x_guess) # this applies our fit function to the entire range defined above

# set up and perform the non-linear regression
mydata3 = odr.RealData(times[1:], fields3adj) # put all our data into a special container
myodr3 = odr.ODR(mydata3, myfunc, beta0 = p03, sstol = 1e-20, job=00000) # set up a fitting data structure
myoutput3 = myodr3.run() # perform the nonlinear regression to find best-fit parameters

sd3  = myoutput3.sd_beta # the normalized standard deviations
p3   = myoutput3.beta # the best-fit parameters

print(p3) # if the fit converged, we should see a list of two parameters: the offset and the slope
print(sd3) # this gives the SD for the model parameters

y_fit3 = f(p3, x_fit) # this applies our fit function to the entire range defined above

plt.plot(times[1:], fields3adj,label='Data')
plt.plot(x_guess, y_guess3,'g',label='Guess')
plt.plot(x_fit, y_fit3,'r',label='Best Fit')
plt.xlabel('Time since switch (s)')
plt.ylabel('Normalized magnetic field (A.U.)')
plt.legend(loc=1)
plt.title(r'$R_s=0.1~\Omega$')
plt.show()

###################################################################################################################
# data for the 200 V Zener diode

voltages4 = []

with open('ALL0004/F0001CH2.csv', 'r', newline='') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    next(csvReader, None)
    for row in csvReader:
        voltages4.append(float(row[4]))

offset4 = np.mean(voltages4[1:20])
voltages4adj = [val - offset4 for val in voltages4]

fields4 = integrate.cumulative_trapezoid(voltages4adj)
fields4adj = [(val -min1)/(max1-min1) for val in fields4]

p04 = [0.0000036, 0.0000712, 1, 14000, .00007] # initial guess for fit parameters: offset, slope

y_guess4 = f(p04, x_guess) # this applies our fit function to the entire range defined above

# set up and perform the non-linear regression
mydata4 = odr.RealData(times[1:], fields4adj) # put all our data into a special container
myodr4 = odr.ODR(mydata4, myfunc, beta0 = p04, sstol = 1e-20, job=00000) # set up a fitting data structure
myoutput4 = myodr4.run() # perform the nonlinear regression to find best-fit parameters

sd4  = myoutput4.sd_beta # the normalized standard deviations
p4   = myoutput4.beta # the best-fit parameters

print(p4) # if the fit converged, we should see a list of two parameters: the offset and the slope
print(sd4) # this gives the SD for the model parameters

y_fit4 = f(p4, x_fit) # this applies our fit function to the entire range defined above

plt.plot(times[1:], fields4adj,label='Data')
plt.plot(x_guess, y_guess4,'g',label='Guess')
plt.plot(x_fit, y_fit4,'r',label='Best Fit')
plt.xlabel('Time since switch (s)')
plt.ylabel('Normalized magnetic field (A.U.)')
plt.legend(loc=1)
plt.title(r'$V_Z=200~V$')
plt.show()

###################################################################################################################
# data for no shunt resistor

voltages5 = []
times5 = []

with open('ALL0005/F0001CH2.csv', 'r', newline='') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=',')
    next(csvReader, None)
    for row in csvReader:
        voltages5.append(float(row[4]))
        times5.append(float(row[3]))

offset5 = np.mean(voltages5[1:20])
voltages5adj = [val - offset5 for val in voltages5]

fields5 = integrate.cumulative_trapezoid(voltages5adj)*2.0 # to adjust for the timebase being different
fields5adj = [(val -min1)/(max1-min1) for val in fields5]

p05 = [0.000006, 0.000022, 1, 7000, .0007] # initial guess for fit parameters: offset, slope

x_guess5 = np.linspace(np.min(times5[1:]), np.max(times5[1:]), 1000) # this defines a range from the minimum of our data to the max of our data, and splits it up into 100 evenly-spaced values
y_guess5 = f(p05, x_guess5) # this applies our fit function to the entire range defined above

# set up and perform the non-linear regression
mydata5 = odr.RealData(times5[1:], fields5adj) # put all our data into a special container
myodr5 = odr.ODR(mydata5, myfunc, beta0 = p05, sstol = 1e-20, job=00000) # set up a fitting data structure
myoutput5 = myodr5.run() # perform the nonlinear regression to find best-fit parameters

sd5  = myoutput5.sd_beta # the normalized standard deviations
p5   = myoutput5.beta # the best-fit parameters

print(p5) # if the fit converged, we should see a list of two parameters: the offset and the slope
print(sd5) # this gives the SD for the model parameters

x_fit5 = np.linspace(np.min(times5[1:]), np.max(times5[1:]), 1000) # this defines a range from the minimum of our data to the max of our data, and splits it up into 100 evenly-spaced values
y_fit5 = f(p5, x_fit5) # this applies our fit function to the entire range defined above

plt.plot(times5[1:], fields5adj,label='Data')
plt.plot(x_guess5, y_guess5,'g',label='Guess')
plt.plot(x_fit5, y_fit5,'r',label='Best Fit')
plt.xlabel('Time since switch (s)')
plt.ylabel('Normalized magnetic field (A.U.)')
plt.legend(loc=1)
plt.title(r'$R_s=0~\Omega$')
plt.show()

###################################################################################################################
# combine all the switching plots together

fig = plt.figure()
plt.plot(times[1:],fields1adj,'r-',label=r'$17~\Omega$')
plt.plot(times[1:],fields2adj,'g-',label=r'$1.0~\Omega$')
plt.plot(times[1:],fields3adj,'b-',label=r'$0.1~\Omega$')
plt.plot(times[1:],fields4adj,'k-',label=r'$200~V$')
plt.plot(times5[1:],fields5adj,'c-',label=r'$0~\Omega$')
plt.xlabel("Time (s)")
plt.ylabel("Field (A.U.)")
plt.suptitle('200 A Field switching')
plt.legend(loc=1)
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()
