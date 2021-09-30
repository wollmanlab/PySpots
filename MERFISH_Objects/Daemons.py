from MERFISH_Objects.Image import *
from MERFISH_Objects.Daemons import *
from MERFISH_Objects.Stack import *
from MERFISH_Objects.Hybe import *
from MERFISH_Objects.Classify import *
from MERFISH_Objects.Segment import *
from MERFISH_Objects.Deconvolution import *
from MERFISH_Objects.Registration import *
from MERFISH_Objects.Position import *
from MERFISH_Objects.Classify import *
from MERFISH_Objects.Segment import *
from MERFISH_Objects.FISHData import *
from MERFISH_Objects.Dataset import *
import threading
import time
import dill as pickle
import os
import multiprocessing
import sys
from tqdm import tqdm
import random
from filelock import Timeout, FileLock

class Class_Daemon(object):
    def __init__(self,directory_path,interval=60,ncpu=10,position=0,verbose=False,error_verbose=False,reverse=False):
        self.reverse = reverse
        self.error_verbose = error_verbose
        self.verbose=verbose
        self.ncpu=ncpu
        self.position = position
        # Set up nessisary directories
        self.directory_path = directory_path
        self.type = directory_path.split('/')[-1]
        if not os.path.exists(self.directory_path):
            os.mkdir(self.directory_path)
        self.input_path = os.path.join(self.directory_path,'input')
        if not os.path.exists(self.input_path):
            os.mkdir(self.input_path)
        self.output_path = os.path.join(self.directory_path,'output')
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.interval = interval

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
        
    def generate_input(self):
        self.input = [i for i in os.listdir(self.input_path) if not 'lock' in i]
        if self.ncpu>1:
            if not np.isin(self.type,['hybe','stack','position','channel']):
                if len(self.input)>5000: # prevents multiprocessing from stuggeling with large numbers (temp fix)
                    if self.verbose:
                        print('Processing 5000 out of '+str(len(self.input)))
                    self.input = np.random.choice(self.input,5000,replace=False)
        if self.reverse:
            self.input.reverse()
            
    def main(self,file):
        input_file = os.path.join(self.input_path,file)
        if 'lock' in file:
            return None
        # Check if it exists
        if not os.path.exists(input_file):
            return None
        # Check File Lock
        if os.path.exists(input_file+'.lock'):
            # Maybe check how old it is
            timeout = 1*60*60 # 1 hour
            if (time.time()-os.path.getmtime(input_file+'.lock'))<timeout:
                # File is in use
                return None
            else:
                # likely file was forgotten about
                lock = FileLock(input_file+'.lock')
        else:
            lock = FileLock(input_file+'.lock')
        try: # nessisary in case a file is currently being written
            with lock:
                class_object = self.generate_class(input_file)
                class_object.run()
                os.remove(input_file+'.lock')
                if class_object.completed:
                    shutil.move(input_file,os.path.join(self.output_path,file))

        except Exception as e:
            try:
                os.remove(input_file+'.lock')
            except:
                pass
            if self.error_verbose:
                print(input_file)
                print('Error:',e,file)
            pass
            return None
            
    def wrapper(self):
        os.environ['MKL_NUM_THREADS'] = '3'
        os.environ['GOTO_NUM_THREADS'] = '3'
        os.environ['OMP_NUM_THREADS'] = '3'
        self.generate_input()
        if len(self.input)>0:
            if self.ncpu>1:
                pool = multiprocessing.Pool(self.ncpu)
                sys.stdout.flush()
                results = pool.imap(self.main, self.input) # might need external function or no bound by module
                if self.verbose:
                    iterable = tqdm(results,total=len(self.input),desc='Class Daemon'+self.directory_path,position=self.position)
                else:
                    iterable = results
                for i in iterable:
                    pass
                pool.close()
                sys.stdout.flush()
            else:
                if self.verbose:
                    iterable = tqdm(self.input,total=len(self.input),desc='Class Daemon'+self.directory_path)
                else:
                    iterable = self.input
                for i in iterable:
                    self.main(i)
            
    def run(self):
        """ Method that runs forever """
        self.finished=False 
        while not self.finished:
            self.wrapper()
            time.sleep(self.interval)
            
    def generate_class(self,fname_path):
        data = pickle.load(open(fname_path,'rb'))
        level = data['level']
        if 'verbose' in data.keys():
            verbose = data['verbose']
        else:
            verbose = False
        if level == 'dataset':
            data_object = Dataset_Class(data['metadata_path'],
                                        data['dataset'],
                                        data['cword_config'],
                                        verbose=verbose)
        elif level == 'position':
            data_object = Position_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['cword_config'],
                                         verbose=verbose)
        elif level == 'hybe':
            data_object = Hybe_Class(data['metadata_path'],
                                     data['dataset'],
                                     data['posname'],
                                     data['hybe'],
                                     data['cword_config'],
                                     verbose=verbose)
        elif level == 'registration':
            data_object = Registration_Class(data['metadata_path'],
                                             data['dataset'],
                                             data['posname'],
                                             data['hybe'],
                                             data['cword_config'],
                                             verbose=verbose)
        elif level == 'stack':
            data_object = Stack_Class(data['metadata_path'],
                                      data['dataset'],
                                      data['posname'],
                                      data['hybe'],
                                      data['channel'],
                                      data['cword_config'],
                                      verbose=verbose)
        elif level == 'deconvolution':
            data_object = Deconvolution_Class(data['metadata_path'],
                                              data['dataset'],
                                              data['posname'],
                                              data['hybe'],
                                              data['channel'],
                                              data['cword_config'],
                                              verbose=verbose)
        elif level == 'image':
            data_object = Image_Class(data['metadata_path'],
                                      data['dataset'],
                                      data['posname'],
                                      data['hybe'],
                                      data['channel'],
                                      data['zindex'],
                                      data['cword_config'],
                                      verbose=verbose)
        elif level == 'segmentation':
            data_object = Segment_Class(data['metadata_path'],
                                             data['dataset'],
                                             data['posname'],
                                             data['cword_config'],
                                             verbose=verbose)
        elif level == 'classification':
            data_object = Classify_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['cword_config'],
                                         verbose=verbose)
        else:
            raise ValueError(level,'Is not implemented')
        return data_object

class Decovolution_Daemon(object):
    def __init__(self,directory_path,interval=60,ncpu=10,position=0,verbose=False,error_verbose=False,reverse=False):
        self.reverse = reverse
        self.error_verbose = error_verbose
        self.verbose=verbose
        self.ncpu=ncpu
        self.position = position
        # Set up nessisary directories
        self.directory_path = directory_path
        if not os.path.exists(self.directory_path):
            os.mkdir(self.directory_path)
        self.input_path = os.path.join(self.directory_path,'input')
        if not os.path.exists(self.input_path):
            os.mkdir(self.input_path)
        self.output_path = os.path.join(self.directory_path,'output')
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.interval = interval

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
        
    def generate_input(self):
        self.input = list(sorted(os.listdir(self.input_path), key=lambda f: os.path.getctime("{}/{}".format(self.input_path, f))))
        if self.reverse:
            self.input.reverse()
        
    def main(self,file):
        input_file = os.path.join(self.input_path,file)
        # File Lock
        if not os.path.exists(input_file):
            return None
        else:
            try: # nessisary in case a file is currently being written
                class_object = self.generate_class(input_file)
                class_object.gpu_algorithm = self.gpu_algorithm
                class_object.run()
                if class_object.completed:
                    shutil.move(input_file,os.path.join(self.output_path,file))
            except Exception as e:
                if self.error_verbose:
                    print(input_file)
                    print('Error:',e,file)
                pass
            return None
            
    def wrapper(self):
        os.environ['MKL_NUM_THREADS'] = '3'
        os.environ['GOTO_NUM_THREADS'] = '3'
        os.environ['OMP_NUM_THREADS'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.generate_input()
        if len(self.input)>0:
            if self.verbose:
                iterable = tqdm(self.input,total=len(self.input),desc='Class Daemon'+self.directory_path,position=self.position)
            else:
                iterable = self.input
            for i in iterable:
                self.main(i)
            
    def run(self):
        """ Method that runs forever """
        self.gpu_algorithm = fd_restoration.RichardsonLucyDeconvolver(3).initialize()
        self.finished=False # Need a way to tell when all of the images that will ever be done are done
        while not self.finished:
            self.wrapper()
            time.sleep(self.interval)
            
    def generate_class(self,fname_path):
        data = pickle.load(open(fname_path,'rb'))
        level = data['level']
        if 'verbose' in data.keys():
            verbose = data['verbose']
        else:
            verbose = False
        if level == 'dataset':
            data_object = Dataset_Class(data['metadata_path'],
                                        data['dataset'],
                                        data['cword_config'],
                                        verbose=verbose)
        elif level == 'position':
            data_object = Position_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['cword_config'],
                                         verbose=verbose)
        elif level == 'hybe':
            data_object = Hybe_Class(data['metadata_path'],
                                     data['dataset'],
                                     data['posname'],
                                     data['hybe'],
                                     data['cword_config'],
                                     verbose=verbose)
        elif level == 'registration':
            data_object = Registration_Class(data['metadata_path'],
                                             data['dataset'],
                                             data['posname'],
                                             data['hybe'],
                                             data['cword_config'],
                                             verbose=verbose)
        elif level == 'stack':
            data_object = Stack_Class(data['metadata_path'],
                                      data['dataset'],
                                      data['posname'],
                                      data['hybe'],
                                      data['channel'],
                                      data['cword_config'],
                                      verbose=verbose)
        elif level == 'deconvolution':
            data_object = Deconvolution_Class(data['metadata_path'],
                                              data['dataset'],
                                              data['posname'],
                                              data['hybe'],
                                              data['channel'],
                                              data['cword_config'],
                                              verbose=verbose)
        elif level == 'image':
            data_object = Image_Class(data['metadata_path'],
                                      data['dataset'],
                                      data['posname'],
                                      data['hybe'],
                                      data['channel'],
                                      data['zindex'],
                                      data['cword_config'],
                                      verbose=verbose)
        elif level == 'segmentation':
            data_object = Segment_Class(data['metadata_path'],
                                             data['dataset'],
                                             data['posname'],
                                             data['cword_config'],
                                             verbose=verbose)
        elif level == 'classification':
            data_object = Classify_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['cword_config'],
                                         verbose=verbose)
        else:
            raise ValueError(level,'Is not implemented')
        return data_object

class Image_Daemon(object):
    def __init__(self,directory_path,interval=60,ncpu=10,position=0,verbose=False,error_verbose=False,reverse=False):
        self.reverse = reverse
        self.error_verbose = error_verbose
        self.verbose=verbose
        self.ncpu=ncpu
        self.position = position
        # Set up nessisary directories
        self.directory_path = directory_path
        if not os.path.exists(self.directory_path):
            os.mkdir(self.directory_path)
        self.input_path = os.path.join(self.directory_path,'input')
        if not os.path.exists(self.input_path):
            os.mkdir(self.input_path)
        self.output_path = os.path.join(self.directory_path,'output')
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.interval = interval

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution
        
    def generate_input(self):
        self.input = list(sorted(os.listdir(self.input_path), key=lambda f: os.path.getctime("{}/{}".format(self.input_path, f))))
        if self.reverse:
            self.input.reverse()
        
    def main(self,file):
        input_file = os.path.join(self.input_path,file)
        # File Lock
        if not os.path.exists(input_file):
            return None
        else:
            try: # nessisary in case a file is currently being written
                class_object = self.generate_class(input_file)
                class_object.gpu_algorithm = self.gpu_algorithm
                class_object.run()
                if class_object.completed:
                    shutil.move(input_file,os.path.join(self.output_path,file))
            except Exception as e:
                if self.error_verbose:
                    print(input_file)
                    print('Error:',e,file)
                pass
            return None
            
    def wrapper(self):
        os.environ['MKL_NUM_THREADS'] = '3'
        os.environ['GOTO_NUM_THREADS'] = '3'
        os.environ['OMP_NUM_THREADS'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.generate_input()
        if len(self.input)>0:
            if self.verbose:
                iterable = tqdm(self.input,total=len(self.input),desc='Class Daemon'+self.directory_path,position=self.position)
            else:
                iterable = self.input
            for i in iterable:
                self.main(i)
            
    def run(self):
        """ Method that runs forever """
        self.gpu_algorithm = fd_restoration.RichardsonLucyDeconvolver(3).initialize()
        self.finished=False # Need a way to tell when all of the images that will ever be done are done
        while not self.finished:
            self.wrapper()
            time.sleep(self.interval)
            
    def generate_class(self,fname_path):
        data = pickle.load(open(fname_path,'rb'))
        level = data['level']
        if 'verbose' in data.keys():
            verbose = data['verbose']
        else:
            verbose = False
        if level == 'dataset':
            data_object = Dataset_Class(data['metadata_path'],
                                        data['dataset'],
                                        data['cword_config'],
                                        verbose=verbose)
        elif level == 'position':
            data_object = Position_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['cword_config'],
                                         verbose=verbose)
        elif level == 'hybe':
            data_object = Hybe_Class(data['metadata_path'],
                                     data['dataset'],
                                     data['posname'],
                                     data['hybe'],
                                     data['cword_config'],
                                     verbose=verbose)
        elif level == 'registration':
            data_object = Registration_Class(data['metadata_path'],
                                             data['dataset'],
                                             data['posname'],
                                             data['hybe'],
                                             data['cword_config'],
                                             verbose=verbose)
        elif level == 'stack':
            data_object = Stack_Class(data['metadata_path'],
                                      data['dataset'],
                                      data['posname'],
                                      data['hybe'],
                                      data['channel'],
                                      data['cword_config'],
                                      verbose=verbose)
        elif level == 'deconvolution':
            data_object = Deconvolution_Class(data['metadata_path'],
                                              data['dataset'],
                                              data['posname'],
                                              data['hybe'],
                                              data['channel'],
                                              data['cword_config'],
                                              verbose=verbose)
        elif level == 'image':
            data_object = Image_Class(data['metadata_path'],
                                      data['dataset'],
                                      data['posname'],
                                      data['hybe'],
                                      data['channel'],
                                      data['zindex'],
                                      data['cword_config'],
                                      verbose=verbose)
        elif level == 'segmentation':
            data_object = Segment_Class(data['metadata_path'],
                                             data['dataset'],
                                             data['posname'],
                                             data['cword_config'],
                                             verbose=verbose)
        elif level == 'classification':
            data_object = Classify_Class(data['metadata_path'],
                                         data['dataset'],
                                         data['posname'],
                                         data['cword_config'],
                                         verbose=verbose)
        else:
            raise ValueError(level,'Is not implemented')
        return data_object
            
