#coding:utf-8

import numpy as np
import os
from subprocess import *
import Stats
from metric import evaluate
from decimal import Decimal, Context, ROUND_HALF_UP

class MyFM():

      temp_dir = '.\\fm\\temp'

      def __init__(self, train_file, test_file, test_file2 = None):
            
            if not os.path.exists(MyFM.temp_dir):
                  os.mkdir(MyFM.temp_dir)

            self.train_file = train_file
            self.test_file = test_file
            self.test_file2 = test_file2

            self.train_tail = os.path.split(self.train_file)[1]
            self.test_tail = os.path.split(self.test_file)[1]

            self.fm_train_exe = r'.\\fm\\libfm.exe'

            assert os.path.exists(self.fm_train_exe), 'train executable not found'

            self.predict_file = os.path.join(MyFM.temp_dir, self.test_tail + '.predict')
            self.result_file = os.path.join(MyFM.temp_dir, self.test_tail + '.result')

            if test_file2:
                  self.test_file2 = test_file2
                  self.test_tail2 = os.path.split(self.test_file2)[1]
                  self.predict_file2 = os.path.join(MyFM.temp_dir, self.test_tail2 + '.predict')
                  self.result_file2 = os.path.join(MyFM.temp_dir, self.test_tail2 + '.result')

            if os.path.exists(self.predict_file):
                  os.remove(self.predict_file)
            if os.path.exists(self.result_file):
                  os.remove(self.result_file)
            if self.test_file2 and os.path.exists(self.predict_file2):
                  os.remove(self.predict_file2)
            if self.test_file2 and os.path.exists(self.result_file2):
                  os.remove(self.result_file2)
                  

      def data_process(self, infile, outfile):
            infile = open(infile)
            data = infile.readlines()
            infile.close()
            
            outfile = open(outfile, 'w')
            outfile.write('labels 1 -1' + '\n')
            for d in data:
                  if float(d.strip()) > 0.5:
                        outfile.write('1 ' + d.strip() + ' ' + str(1 - float(d.strip())) + '\n')
                  else:
                        outfile.write('-1 ' + d.strip() + ' ' + str(1 - float(d.strip())) + '\n')
            outfile.close()

                  
      
      def train(self):
#=========================================================================MCMC=============================================================================
            command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method mcmc -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file)
            Popen(command_train, shell = True, stdout = PIPE).communicate()

            if self.test_file2:
                  command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method mcmc -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2)
                  Popen(command_train, shell = True, stdout = PIPE).communicate()
#===========================================================================SGD============================================================================
            # command_train = "{0} -task c -verbosity 0 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgd -learn_rate 0.01 -regular '0,0,0.01' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file)
            # Popen(command_train, shell = True, stdout = PIPE).communicate()

            # if self.test_file2:
            #       command_train = "{0} -task c -verbosity 0 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgd -learn_rate 0.01 -regular '0,0,0.01' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2)
            #       Popen(command_train, shell = True, stdout = PIPE).communicate()
#============================================================================ALS===========================================================================
            # command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method als -regular '0,0,10' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file)
            # Popen(command_train, shell = True, stdout = PIPE).communicate()

            # if self.test_file2:
            #       command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method als -regular '0,0,10' -init_stdev 0.1".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2)
            #       Popen(command_train, shell = True, stdout = PIPE).communicate()
#==============================================================================SGDA========================================================================
            # command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgda -learn_rate 0.01 -init_stdev 0.1 -validation {4}".format(self.fm_train_exe, self.train_file, self.test_file, self.predict_file, self.train_file)
            # Popen(command_train, shell = True, stdout = PIPE).communicate()

            # if self.test_file2:
            #       command_train = "{0} -task c -verbosity 1 -train {1} -test {2} -out {3} -dim '1,1,8' -iter 1000 -method sgda -learn_rate 0.01 -init_stdev 0.1 -validation {4}".format(self.fm_train_exe, self.train_file, self.test_file2, self.predict_file2, self.train_file)
            #       Popen(command_train, shell = True, stdout = PIPE).communicate()
            

      def predict(self):
            self.data_process(self.predict_file, self.result_file)

            if self.test_file2:
                  self.data_process(self.predict_file2, self.result_file2)


      def evaluat(self):
            if self.test_file2:
                  in_file = open(self.test_file2)
                  gold_data = [float(line.strip().split()[0]) for line in in_file]
                  in_file.close()

                  in_file = open(self.result_file2)
                  classifier_data = [float(line.strip().split()[0]) for line in in_file.readlines()[1:]]
                  in_file.close()
                  p, r, f, auc1 = evaluate(np.asarray(gold_data), np.asarray(classifier_data))

                  return (p, r, f, auc1)
            else:
                  in_file = open(self.test_file)
                  gold_data = [float(line.strip().split()[0]) for line in in_file]
                  in_file.close()

                  in_file = open(self.result_file)
                  classifier_data = [float(line.strip().split()[0]) for line in in_file.readlines()[1:]]
                  in_file.close()
                  p, r, f, auc1 = evaluate(np.asarray(gold_data), np.asarray(classifier_data))

                  return (p, r, f, auc1)


if __name__ == '__main__':
      with open('demo.train', "r") as f: # training set
            with open('./fm/temp/train_fm.txt', "w") as f1:
                  for line in f:
                        items = line.strip().split()
                        if int(items[0]) == 1:
                              f1.write("1" + " ")
                              for i in xrange(1, len(items)):
                                    f1.write(items[i] + " ")
                              f1.write("\n")
                        if int(items[0]) == 2:
                              f1.write("-1" + " ")
                              for i in xrange(1, len(items)):
                                    f1.write(items[i] + " ")
                              f1.write("\n")
                        if int(items[0]) == -1:
                              f1.write("-1" + " ")
                              for i in xrange(1, len(items)):
                                    f1.write(items[i] + " ")
                              f1.write("\n")   
      f.close()
      f1.close()
      with open('demo.test', "r") as f: # test set
            with open('./fm/temp/test_fm.txt', "w") as f1:
                  for line in f:
                        items = line.strip().split()
                        if int(items[0]) == 1:
                              f1.write("1" + " ")
                              for i in xrange(1, len(items)):
                                    f1.write(items[i] + " ")
                              f1.write("\n")
                        if int(items[0]) == 2:
                              f1.write("-1" + " ")
                              for i in xrange(1, len(items)):
                                    f1.write(items[i] + " ")
                              f1.write("\n")
                        if int(items[0]) == -1:
                              f1.write("-1" + " ")
                              for i in xrange(1, len(items)):
                                    f1.write(items[i] + " ")
                              f1.write("\n") 
      f.close()
      f1.close()

      if os.path.exists('common.txt'):
          os.remove('common.txt')
      
      total_line = 0
      with open('./fm/temp/train_fm.txt', "r") as f:
          with open('./fm/temp/test_fm.txt', "r") as f1:
              with open('common.txt', "a") as f2:
                  for line in f:
                      total_line += 1
                      f2.write(line)
                  for line in f1:
                      total_line += 1
                      f2.write(line)
      f.close()
      f1.close()
      f2.close()
      
      def build_data_cv(common_file, cv=5):
          revs = []
          with open(common_file, "r") as f:
              for line in f:
                  datum = {
                           "line": line.strip(),
                           "split": np.random.randint(0,cv)
                          }
                  revs.append(datum)
          return revs
          
      def build_train_test_cv(revs, cv):
          train, test = [], []
          for rev in revs:
              if rev["split"]==cv:
                  test.append(rev["line"])
              else:
                  train.append(rev["line"])
          np.random.seed(3435)
          train = np.random.permutation(train)
          test = np.random.permutation(test)
          with open('./fm/temp/train.txt', "w") as f:
              for item in train:
                  f.write(item + "\n")
          f.close()
          with open('./fm/temp/test.txt', "w") as f1:
              for item in test:
                  f1.write(item + "\n")
          f1.close()
      
      revs = build_data_cv("common.txt")
      
      P_value = []
      R_value = []
      F1_value = []
      auc1_value = []
      
      r = range(5)
      for i in r:
          build_train_test_cv(revs, i)
          print "cv%d:" % i
          l = MyFM('./fm/temp/train.txt', './fm/temp/test.txt')
          l.train()
          l.predict()
          p, r, f, auc1 = l.evaluat()
          P_value.append(p)
          R_value.append(r)
          F1_value.append(f)
          auc1_value.append(auc1)
      
      stats_P = Stats.Stats(P_value)
      stats_R = Stats.Stats(R_value)
      stats_F1 = Stats.Stats(F1_value)
      stats_auc1 = Stats.Stats(auc1_value)
      
      print "P_value:", P_value
      print "R_value:", R_value
      print "F1_value:", F1_value
      print "auc_value:", auc1_value
      
      print 'stats_P(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_P.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_P.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_R(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_R.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_R.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_F1(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_F1.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_F1.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
      print 'stats_auc(mean value ± standard deviation): %.2f±%.2f' % (Decimal(str(stats_auc1.avg())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)), Decimal(str(stats_auc1.stdev())).normalize(Context(prec=3, rounding=ROUND_HALF_UP)))
