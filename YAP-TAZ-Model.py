#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pysb import *
from pysb.integrate import Solver 
from pysb.simulator import ScipyOdeSimulator
import matplotlib.pyplot as plt
import numpy as np
import itertools
import sympy
#Define Model
Model()


# In[2]:


# Set Monomers (___ <---We can input any value we want in here)

Monomer('FAK', ['state'], {'state': ['p', 'u']})
Monomer('RhoA', ['state'], {'state': ['gtp', 'gdp']})
Monomer('ROCK', ['state'], {'state': ['a', 'i']})

Monomer('mDia', ['state'], {'state': ['A', 'i']})
Monomer('Myo', ['state'], {'state': ['A', 'i']})
Monomer('LIMK', ['state'], {'state': ['A', 'i']})
Monomer('Cofilin', ['state'], {'state': ['np', 'p']})
Monomer('G_actin', ['state'], {'state': ['Factin', 'i']})

Monomer('YAPTAZ', ['state'], {'state': ['i', 'p', 'nuc']})
Monomer('LaminA', ['state'], {'state': ['p', 'i']})
Monomer('NPC', ['state'], {'state': ['A', 'i']})


# In[3]:


# Setting Rate Constants

# Other Constants
    #V is in Liters and we Choose what E is
V = (4/3*(3.14159265)*(50e-6)**3)*1000
Avog_num = 6.022e23
# Units are in um^2
cell_SA = 4*(3.14159265)*(50)**2

pFAK = []
FAK = []
RhoAGTP = []
MyoA = []
Myo = []
LaminA = []
LaminAp = []
YAPTAZnuc = []
YAPTAZcyto = []

stiffness = np.arange(0,8.0, 0.1)
for expo in stiffness

    E = 10^expo
    # FAK Reaction

    # Original Units for all (1/s)
    Parameter('kf', 0.015)
    Parameter('kdf', 0.035)
    Parameter('ksf', 0.379*(E/(E+3.250)))
    # RhoA Reaction

    # Original Units: (1/s)
    Parameter('kfkp', 0.0168)
    #Original Units: 1/(uM^5)
    Parameter('gamma', 77.56*1e30/((Avog_num*V)**5))
    #Original Units: unitless
    Parameter('n', 5)
    # Used for 1st order Reverse reaction
    # Original Units: (1/s)
    Parameter('kde', 0.625)

    # ROCK Reaction

    # Will be used for 2nd order reaction
    # Original units: 1/(s*uM)
    Parameter('krp', 0.648*1e6/(V*Avog_num))
    # Will be used for 1st order reaction
    # Original Units: (1/s)
    Parameter('kd', 0.8)

    # mDia Reaction

    # Used for 2nd order reactions
    # Original Units: 1/(s*uM)
    Parameter('kmp', 0.002*1e6/(V*Avog_num))
    # Used for 1st order reactions
    # Original Units: 1/s
    Parameter('kdmDia', 0.005)

    # Myo Reaction

    # Both are 1st order
    # Original Units: 1/s
    Parameter('kmr', 0.03)
    # Original Units: 1/uM
    Parameter('e', 36*1e6/(Avog_num*V))
    # Original Units: 1/s
    Parameter('kdmy', 0.067)

    # LIMK Reaction

    # Both are 1st order
    # Original Units: 1/s
    Parameter('klr', 0.07)
    # Original Units: 1/uM
    Parameter('tau', 55.49*1e6/(Avog_num*V))
    # Original Units: 1/s
    Parameter('kdl', 2)

    # Cofilin Reaction

    # Used for 1st order reaction
    # Original Units: 1/s
    Parameter('kturnover', 0.04)
    # Used for 1st order reaction
    # Original Units: 1/s
    Parameter('kcatcofilin', 0.34)
    Parameter('kmcofilin', 4*1e-6*V*Avog_num)

    # G-actin to Factin Reaction

    # Used for 1st order reaction
    # Original Units: 1/s
    Parameter('kra', 0.4)
    # Original Units: 1/uM
    Parameter('alpha', 50*1e6/(Avog_num*V))
    # Used for 1st order reaction
    # Original Units: 1/s
    Parameter('kdep', 3.5)
    # Used for 2nd order reaction
    # Original Units: 1/(s*uM)
    Parameter('kfc1', 4*1e6/(V*Avog_num))

    # For the tanh Functions:
    # Original Units: 1/(uM)
    Parameter('sc1', 20*1e6/(V*Avog_num))
    # Original Units: uM
    Parameter('ROCKB', 0.3*1e-6*V*Avog_num)
    Parameter('mDiaB', 0.165*1e-6*V*Avog_num)

    # YAPTAZ Reactions

    # Used for 1st order reactions
    # Original Units: 1/s
    Parameter('knc', 0.14)
    Parameter('kcn', 0.56)
    # Used for 1st order reactions (although these are kinda weird)
    # Original Units: #/(s*uM*um^2)
    Parameter('kinb', 1*1e6*cell_SA/(Avog_num*V)) # <---- Derived using some calculations
    Parameter('kout', 1*1e6*cell_SA/(Avog_num*V))
    # Used for 3rd order reactions
    # Original Units 1/((uM*)2)*s)
    Parameter('kcy', 7.6e-4*1e12/((V*Avog_num)**2))
    # Used for 2nd order Reaction
    # Original Units: 1/(uM*s)
    Parameter('kin', 10*1e6/(V*Avog_num))

    # LaminA Reactins

    # Used for 1st order reactions
    # Original Units: 1/s
    Parameter('krl', 0.001)
    Parameter('kfl', 0.46)
    # Original Units: kPa/(uM^2.6)
    Parameter('p', 9e-6)

    # NPC Reactions

    # Used for 1st order reactions
    # Original Units: 1/s
    Parameter('kr', 8.7)
    # Used for a 4th order reaction
    #Original Units: um^2/(#*uM^2*s)
    Parameter('kfnpc', 2.8e-7*1e12/(cell_SA*(V*Avog_num)**2))




    # In[4]:


    # Setting the Initial Values

    Parameter('FAKp_0', 0.3*1e-6*V*Avog_num)
    Initial(FAK(state = 'p'), FAKp_0)
    Parameter('FAKu_0', 0.7*1e-6*V*Avog_num)
    Initial(FAK(state = 'u'), FAKu_0)

    Parameter('RhoAgtp_0', 33.6*cell_SA)
    Initial(RhoA(state = 'gtp'), RhoAgtp_0)
    Parameter('RhoAgdp_0', 1*1e-6*V*Avog_num)
    Initial(RhoA(state = 'gdp'), RhoAgdp_0 )

    Parameter('ROCKa_0',0)
    Initial(ROCK(state = 'a'), ROCKa_0)
    Parameter('ROCKi_0', 1*1e-6*V*Avog_num)
    Initial(ROCK(state = 'i'), ROCKi_0)

    Parameter('mDiaA_0', 0)
    Initial(mDia(state = 'A'), mDiaA_0)
    Parameter('mDiai_0', 0.8*1e-6*V*Avog_num)
    Initial(mDia(state = 'i'), mDiai_0)

    Parameter('MyoA_0', 1.5*1e-6*V*Avog_num)
    Initial(Myo(state = 'A'), MyoA_0)
    Parameter('Myoi_0', 3.5*1e-6*V*Avog_num)
    Initial(Myo(state = 'i'), Myoi_0)

    Parameter('LIMKA_0', 0.1*1e-6*V*Avog_num)
    Initial(LIMK(state = 'A'), LIMKA_0)
    Parameter('LIMKi_0', 1.9*1e-6*V*Avog_num)
    Initial(LIMK(state = 'i'), LIMKi_0)

    Parameter('CofilinNP_0', 1.8*1e-6*V*Avog_num)
    Initial(Cofilin(state = 'np'), CofilinNP_0 )
    Parameter('CofilinP_0', 0.2*1e-6*V*Avog_num)
    Initial(Cofilin(state = 'p'), CofilinP_0 )

    Parameter('Factin_0', 17.9*1e-6*V*Avog_num)
    Initial(G_actin(state = 'Factin'), Factin_0)
    Parameter('G_actin_0', 482.4*1e-6*V*Avog_num)
    Initial(G_actin(state = 'i'), G_actin_0)

    Parameter('YAPTAZi_0', 0.7*1e-6*V*Avog_num)
    Initial(YAPTAZ(state = 'i'), YAPTAZi_0)
    Parameter('YAPTAZnuc_0', 0.7*1e-6*V*Avog_num)
    Initial(YAPTAZ(state = 'nuc'), YAPTAZnuc_0)
    Parameter('YAPTAZp_0', 0.2*1e-6*V*Avog_num)
    Initial(YAPTAZ(state = 'p'), YAPTAZp_0)

    Parameter('LaminAp_0', 3500*cell_SA)
    Initial(LaminA(state = 'p'), LaminAp_0)
    Parameter('LaminAi_0', 0)
    Initial(LaminA(state = 'i'), LaminAi_0)

    Parameter('NPCA_0', 0)
    Initial(NPC(state = 'A'), NPCA_0)
    Parameter('NPCi_0', 6.5*cell_SA)
    Initial(NPC(state = 'i'), NPCi_0)


    # In[5]:


    # Observables
    Observable('obsFAKp', FAK(state = 'p'))
    Observable('obsFAKu', FAK(state = 'u'))

    Observable('obsRhoAgtp', RhoA(state = 'gtp'))
    Observable('obsRhoAgdp', RhoA(state = 'gdp'))

    Observable('obsROCKa', ROCK(state = 'a'))
    Observable('obsROCKi', ROCK(state = 'i'))

    Observable('obsmDiaA', mDia(state = 'A'))
    Observable('obsmDiai', mDia(state = 'i'))

    Observable('obsMyoA', Myo(state = 'A'))
    Observable('obsMyoi', Myo(state = 'i'))

    Observable('obsLIMKA',LIMK(state = 'A'))
    Observable('obsLIMKi',LIMK(state = 'i'))

    Observable('obsCofilinNP', Cofilin(state = 'np'))
    Observable('obsCofilinP', Cofilin(state = 'p'))

    Observable('obsFactin', G_actin(state = 'Factin'))
    Observable('obsG_actin', G_actin(state = 'i'))

    Observable('obsYAPTAZnuc', YAPTAZ(state = 'nuc'))
    Observable('obsYAPTAZp', YAPTAZ(state = 'p'))
    Observable('obsYAPTAZi', YAPTAZ(state = 'i'))

    Observable('obsLaminAp', LaminA(state = 'p'))
    Observable('obsLaminAi', LaminA(state = 'i'))

    Observable('obsNPCA', NPC(state = 'A'))
    Observable('obsNPCi', NPC(state = 'i'))


    # In[6]:


    # Rules

    # FAK
    # Regular Forward and Backward FAK Reaction
    Rule('activ_FAK', FAK(state = 'u') | FAK(state = 'p'), kf, kdf)
    # Forward FAK Reaction activated by stiffness
    Rule('activ_FAK_stiff', FAK(state = 'u') >> FAK(state = 'p'), ksf)

    # RhoA
    Expression('RhoAgdp_to_RhoAgtp', kfkp*(gamma*obsFAKp**5+1))
    Rule('activ_RhoA', RhoA(state = 'gdp') | RhoA(state = 'gtp'), RhoAgdp_to_RhoAgtp, kde)

    # ROCK
    # 2nd Order Forward Reaction
    Rule('activ_ROCK', ROCK(state = 'i') + RhoA(state = 'gtp') >> ROCK(state = 'a') + RhoA(state = 'gtp'), krp)
    # 1st Order Reverse Reaction
    Rule('deactiv_ROCK', ROCK(state = 'a') >> ROCK(state = 'i'), kd)

    # mDia
    # 2nd Order Forward Reaction
    Rule('activ_mDia', RhoA(state = 'gtp') + mDia(state = 'i') >> RhoA(state = 'gtp') + mDia(state = 'A'), kmp)
    # 1st Order Reverse Reaction
    Rule('deactiv_mDia', mDia(state = 'A') >> mDia(state = 'i'), kdmDia)

    # Myo
    Expression('Myo_to_MyoA', kmr*(e*((sympy.tanh(sc1*(obsROCKa - ROCKB)) + 1)*obsROCKa*0.5) +1 ))
    Rule('activ_Myo', Myo(state = 'i') | Myo(state = 'A'), Myo_to_MyoA, kdmy)

    # LIMK
    Expression('LIMK_to_LIMKA', klr*(tau*((sympy.tanh(sc1*(obsROCKa - ROCKB)) + 1)*obsROCKa*0.5) +1 ))
    Rule('activ_LIMKA', LIMK(state = 'i') | LIMK(state = 'A'), LIMK_to_LIMKA, kdl)

    # Cofilin
    Rule('activ_cofilin', Cofilin(state = 'p') >> Cofilin(state = 'np'), kturnover)
    Expression('cofilinNP_to_cofilinP', kcatcofilin/(kmcofilin + obsCofilinNP))
    Rule('deactiv_cofilin', Cofilin(state = 'np') + LIMK(state = 'A') >> Cofilin(state = 'p') + LIMK(state = 'A'), cofilinNP_to_cofilinP)

    # G_Actin to Factin
    Expression('G_actin_to_Factin', kra*(alpha*((sympy.tanh(sc1*(obsmDiaA - mDiaB)) + 1)*obsmDiaA*0.5) +1 ))
    Rule('activ_G_actin', G_actin(state = 'i') | G_actin(state = 'Factin'), G_actin_to_Factin, kdep)
    Rule('deactiv_G_actin_Cofilin', G_actin(state = 'Factin') + Cofilin(state = 'np') >> G_actin(state = 'i') + Cofilin(state = 'np'), kfc1)

    # YAPTAZ
    # Basal YAPTAZ to YAPTAZp
    Rule('YAPTAZ_to_YAPTAZp_basal', YAPTAZ(state = 'i') | YAPTAZ(state = 'p'), knc, kcn)
    # YAPTAZp to YAPTAZ Catalyzed by Factin and MyoA
    Rule('YAPTAZp_to_YAPTAZ', G_actin(state = 'Factin') + Myo(state = 'A') + YAPTAZ(state = 'p') >> G_actin(state = 'Factin') + Myo(state = 'A') + YAPTAZ(state = 'i'), kcy)
    # YAPTAZnuc to YAPTAZ Basal
    Rule('YAPTAZnuc_to_YAPTAZ', YAPTAZ(state = 'i') | YAPTAZ(state = 'nuc'), kinb, kout)
    # YAPTAZ to YAPTAZnuc with NPCA opening
    Rule('YAPTAZ_to_YAPTAZnuc', NPC(state = 'A') + YAPTAZ(state = 'i') >> NPC(state = 'A') + YAPTAZ(state = 'nuc'), kin )

    # LaminA
    Expression('Stiffness_LaminA', (p*(obsFactin*1e6/(V*Avog_num))**2.6)/(100 +(p*(obsFactin*1e6/(V*Avog_num))**2.6)))
    Expression('LaminA_Deactiv', Stiffness_LaminA*kfl)
    Rule('LaminA_to_LaminAp', LaminA(state = 'i') | LaminA(state = 'p'), krl, LaminA_Deactiv)

    # NPC
    # NPC Closing
    Rule('NPC_close', NPC(state = 'A') >> NPC(state = 'i'), kr)
    # NPC Opening
    Rule('NPC_open', NPC(state = 'i') + LaminA(state = 'i') + G_actin(state = 'Factin') + Myo(state = 'A') >> NPC(state = 'A') + LaminA(state = 'i') + G_actin(state = 'Factin') + Myo(state = 'A'), kfnpc)


    # In[7]:


    # for i,s in enumerate(model.species):
    #    print(i,s)
    #print()
    #for a,r in enumerate(model.reactions):
       # print(a,r['rate'])
    #print()
    # for i, ode in enumerate(model.odes):
    #     print(i,'ds%d/dt = '%i,ode)

    tspan = np.linspace(0, 100000, 500)
    sim = ScipyOdeSimulator(model, tspan)
    result = sim.run()

    pFAKval = result.observables['obsFAKp'][-1]
    FAKval = result.observables['obsFAKu'][-1]
    RhoAGTPval = result.observables['obsRhoAgtp'][-1]
    MyoAval = result.observables['obsMyoA'][-1]
    Myoval = result.observables['obsMyoi'][-1]
    LaminAval = result.observables['obsLaminAi'][-1]
    LaminApval = result.observables['obsLaminAp'][-1]
    YAPTAZnucval = result.observables['obsYAPTAZnuc'][-1]
    YAPTAZcytoval = result.observables['obsYAPTAZp'][-1] + result.observables['obsYAPTAZi'][-1]

    pFAK.append(pFAKval)
    FAK.append(FAKval)
    RhoAGTP.append(RhoAGTPval)
    MyoA.append(MyoAval)
    Myo.append(Myoval)
    LaminA.append(LaminAval)
    LaminAp.append(LaminApval)
    YAPTAZnuc.append(YAPTAZnucval)
    YAPTAZcyto.append(YAPTAZcytoval)

FAK_tot = np.array(pFAK) + np.array(FAK)
Myo_tot = np.array(MyoA) + np.array(Myo)

pFAK_FAKtot = np.divide(pFAK, FAK_tot)
MyoA_Myotot = np.divide(MyoA, Myo_tot)
LaminA_LaminAp = np.divide(LaminA, LaminAp)
YAPTACnuc_cyto = np.divide(YAPTAZnuc, YAPTAZcyto)
RhoAGTP_convert = (1e6/(Avog_num*cell_SA))*np.array(RhoAGTP)

plt.title('Ratio of pFAK to Total FAK')
plt.plot(stiffness, pFAK_FAKtot)
plt.xlabel('log(kPa)')
plt.ylabel('pFAK/Total FAK')
plt.show()

plt.title('Ratio of MyoA to Total Myo')
plt.plot(stiffness, MyoA_Myotot)
plt.xlabel('log(kPa)')
plt.ylabel('MyoA/Total Myo')
plt.show()

plt.title('Ratio of Nuclear YAPTAZ to Cytosolic YAPTAZ')
plt.plot(stiffness, YAPTACnuc_cyto)
plt.xlabel('log(kPa)')
plt.ylabel('YAPTAZ N/C')
plt.show()

plt.title('RhoA GTP')
plt.plot(stiffness, LaminA_LaminAp)
plt.xlabel('log(kPa)')
plt.ylabel('RhoAGTP (umol/um^2)')
plt.show()

plt.title('Ratio of LaminA to LaminAp')
plt.plot(stiffness, LaminA_LaminAp)
plt.xlabel('log(kPa)')
plt.ylabel('LaminA/LaminAp')
plt.show()

# plt.title('FAK and FAKp')
# plt.plot(tspan, result.observables['obsFAKu'], tspan, result.observables['obsFAKp'])
# plt.legend(['FAK', 'FAKp'])
# plt.show()

# plt.title('RhoAGDP and RhoAGTP')
# plt.plot(tspan, result.observables['obsRhoAgdp'], tspan, result.observables['obsRhoAgtp'])
# plt.legend(['RhoAGDP', 'RhoAGTP'])
# plt.show()

# plt.title('ROCK and ROCKa')
# plt.plot(tspan, result.observables['obsROCKi'], tspan, result.observables['obsROCKa'])
# plt.legend(['ROCK', 'ROCKa'])
# plt.show()

# plt.title('mDia and mDiaA')
# plt.plot(tspan, result.observables['obsmDiai'], tspan, result.observables['obsmDiaA'])
# plt.legend(['mDia', 'mDiaA'])
# plt.show()

# plt.title('Myo and MyoA')
# plt.plot(tspan, result.observables['obsMyoi'], tspan, result.observables['obsMyoA'])
# plt.legend(['Myo', 'MyoA'])
# plt.show()

# plt.title('LIMK and LIMKA')
# plt.plot(tspan, result.observables['obsLIMKi'], tspan, result.observables['obsLIMKA'])
# plt.legend(['LIMK', 'LIMKA'])
# plt.show()

# plt.title('CofilinP and CofilinNP')
# plt.plot(tspan, result.observables['obsCofilinP'], tspan, result.observables['obsCofilinNP'])
# plt.legend(['CofilinP', 'CofilinNP'])
# plt.show()

# plt.title('G-actin and Factin')
# plt.plot(tspan, result.observables['obsG_actin'], tspan, result.observables['obsFactin'])
# plt.legend(['G-actin', 'Factin'])
# plt.show()

# plt.title('YAPTAZ, YAPTAZp and YAPTAZnuc')
# plt.plot(tspan, result.observables['obsYAPTAZi'], tspan, result.observables['obsYAPTAZp'], tspan, result.observables['obsYAPTAZnuc'])
# plt.legend(['YAPTAZ', 'YAPTAZp', 'YAPTAZnuc'])
# plt.show()

# plt.title('LaminA and LaminAp')
# plt.plot(tspan, result.observables['obsLaminAi'], tspan, result.observables['obsLaminAp'])
# plt.legend(['LaminA', 'LaminAp'])
# plt.show()

# plt.title('NPC and NPCA')
# plt.plot(tspan, result.observables['obsNPCi'], tspan, result.observables['obsNPCA'])
# plt.legend(['NPC', 'NPCA'])
# plt.show()










