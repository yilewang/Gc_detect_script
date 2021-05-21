

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
import time
import logging



if __name__ == "__main__":
    grp_pools = ['AD', 'MCI','NC','SNC']
    start = time.time()
    pdList = []
    for grp in grp_pools[0:1]:
        pth = 'C:/Users/Wayne/output/'+grp
        case_pools = os.listdir(pth)
        for caseid in case_pools[0:1]:
            gRange = np.round(np.arange(0.001, 0.08, 0.001), 3)
            gMax = []
            gC = []
            for gm in gRange[0:25]:
                # G critical
                dataFile = 'C:/Users/Wayne/tvb/LFP/'+grp+'/'+caseid+'/'+caseid+'_'+str(gm)+'.csv'
                df = pd.read_csv(dataFile, index_col=0)
                df1 = df.iloc[:, 0:4]
                df2 = df.iloc[:, 6:16]
                df_ex = pd.concat([df1, df2], axis=1)
                dfPCG = df.iloc[:, 4:6]
                avgPCG = np.average(dfPCG, axis =1)
                varPCG = np.var(dfPCG, axis = 1)
                avgRest = np.average(df_ex, axis = 1)
                varRest = np.var(df_ex, axis=1)
                yPCG = avgPCG +varPCG
                yPCG_ = avgPCG - varPCG
                yRest = avgRest + varRest
                yRest_ = avgRest - varRest
                t = np.arange(0, 81920, 1)
                peaksPCG, _ = find_peaks(yPCG, prominence = 1.96*np.std(yPCG))
                peaksRest, _ = find_peaks(yRest, prominence=1.96*np.std(yRest))
                peaksPCG = peaksPCG[peaksPCG>10000]
                peaksRest = peaksRest[peaksRest>10000]
                if len(peaksPCG) > 0 and len(peaksRest) <1:
                    critical = 1
                else:
                    critical = 0
                gC.append(critical)


                # G max
                dataFile = 'C:/Users/Wayne/tvb/LFP/'+grp+'/'+caseid+'/'+caseid+'_'+str(gm)+'.csv'
                df = pd.read_csv(dataFile, index_col=0)
                avg = np.average(df, axis = 1)
                var = np.var(df, axis=1)
                y = avg + var
                y_ = avg - var
                t = np.arange(0, 81920, 1)
                peaks, _ = find_peaks(y, prominence= 1.96* np.std(y))
                peaks = peaks[peaks>10000]
                if len(peaks) > 0:
                    max = 1
                else:
                    max = 0
                gMax.append(max)

                fig, (axs1,axs2,axs3) = plt.subplots(3, figsize=(15,8))
                plt.suptitle(grp+"_"+caseid+'_'+str(gm))
                axs1.fill_between(t, y, y_)
                axs1.plot(t, avg, 'r')
                axs1.plot(peaks, y[peaks], 'xk', label = "ALL Regions")
                axs1.legend()                 
                axs2.fill_between(t, yRest, yRest_)
                axs2.plot(t, avgRest, 'r')
                axs2.plot(peaksRest, yRest[peaksRest], '*r', label = "Rest of Regions")
                axs2.legend()
                axs3.fill_between(t, yPCG, yPCG_)
                axs3.plot(t, avgPCG, 'r')
                axs3.plot(peaksPCG, yPCG[peaksPCG], 'og', label = "PCG")
                axs3.legend()
                # plt.show()
                save_path = grp+"_"+caseid+"_"+str(gm)+"_demo.png"
                plt.savefig(save_path)
            end = time.time()
            logging.warning('Duration: {}'.format(end - start))

            # plt.figure(figsize=(18,5))
            # plt.title(grp+"_"+caseid+'_'+'bifurcation')
            # plt.plot(gRange,gMax, "o:b", label = "Gmax")
            # plt.plot(gRange, gC, "*:r", label = "Gcritical")
            # plt.xticks(np.arange(0.001, 0.080, 0.001))
            # plt.legend()
            # save_path = grp+"_"+caseid+"_"+"bif.png"
            # plt.savefig(save_path)
    #         for cc in range(len(gRange)):
    #             if gMax[cc] - gC[cc] == 0.0:
    #                 res = cc+1
    #                 break
    #         pdList.append((grp, caseid, np.round((res)*0.001,3), np.sum(gMax)*0.001))
    # dfTable = pd.DataFrame(pdList, columns=('Group', 'CaseID', 'Gc', 'Gmax'))
    # dfTable.to_csv(r'C:/Users/Wayne/tvb/TVB_workflow/new_g_max/New_Gc_Gmax_Table.csv', index = False) 

