"""
This is a set of useful features for reading a writing
some basic lammps output files

###############################################################################
###############################################################################
author: Aaron Schwan
email: schwanaaron@gmail.com
github: https://github.com/AaronSchwan
###############################################################################
###############################################################################

"""




#default imports
import sys
import os
import gc
import ntpath
import pickle
import time

#non-default imports
import pandas as pd


#custom imports
import basicFunctions as bf


class dumpFile():
    """
    This will read a file path that is either a pickle file or a lammps dump
    file

    obj = dumpFile(file_path)

    valid calls:
    obj.timestep = returns timestep in the file
    obj.numberofatoms = numbers of atoms in the dump
    obj.boxbounds = returns the bounds with type,low,high in a pandas datframe
    obj.atoms = atomic data


    """

    def __init__(self,file_path):
        file_name = ntpath.basename(file_path).split(".")
        if file_name[len(file_name)-1] == "pkl":
            values = self.bianary_dump(file_path)
        else:
            values = self.lammps_dump(file_path)

        self.timestep = values[0]
        self.numberofatoms = values[1]
        self.boxbounds = values[2]
        self.atoms = values[3]


    def lammps_dump(self,file_path):
        raw_data = pd.read_csv(file_path)

        titles = raw_data.iloc[7].str.split(expand = True).iloc[0][2:].tolist()
        data = pd.DataFrame(raw_data.iloc[8:,0].str.split(' ',len(titles)-1).tolist(), columns = titles)
        data =  data.apply(pd.to_numeric)


        boxboundtype =  raw_data.iloc[3,0].replace("ITEM: BOX BOUNDS ","").split(" ")
        boxboundings = pd.DataFrame(raw_data.iloc[4:7,0].str.split(' ',1).tolist())
        box_dict = {"labels":["type","low","high"],"x":[boxboundtype[0],boxboundings.iloc[0,0],boxboundings.iloc[0,1]],"y":[boxboundtype[1],boxboundings.iloc[1,0],boxboundings.iloc[1,1]],"z":[boxboundtype[2],boxboundings.iloc[2,0],boxboundings.iloc[2,1]]}

        return int(raw_data.iloc[0,0]),int(raw_data.iloc[2,0]),pd.DataFrame.from_dict(box_dict).set_index("labels"),data

    def bianary_dump(self,file_path):
        with open(file_path, 'rb') as input:
            data = pickle.load(input)
            return data.timestep,data.numberofatoms,data.boxbounds,data.atoms

class file():
    """
    This will read a file either from a path to a lammps out files from fix in
    format of standard or pickle

    file format:# Fix*********
                # column ids....
                columnized data

    obj = file(file_path)

    valid calls:
    obj.data

    """

    def __init__(self,file_path):
        file_name = ntpath.basename(file_path).split(".")
        if file_name[len(file_name)-1] == "pkl":
            self.data = self.bianary_data(file_path)
        else:
            self.data = self.lammps_data(file_path)

    def lammps_data(self,file_path):
        raw_data = pd.read_csv(file_path)
        titles = raw_data.iloc[0].str.split(expand = True).iloc[0][1:].tolist()
        data = pd.DataFrame(raw_data.iloc[1:,0].str.split(' ',len(titles)-1).tolist(), columns = titles)
        data =  data.apply(pd.to_numeric)

        return data

    def bianary_data(self,file_path):
        with open(file_path, 'rb') as input:
            data = pickle.load(input).data
        return data

#END OF CLASS DEFINITIONS#######################################################



def dump_class_to_lammps(dump_class,file_path):
    """
    This will write a class from dumpFile() class and convert it from a class
    to a ovito/lammps readable file this is good for appending columns to a
    the obj.atoms data

    dump_class_to_lammps(dump_class,file_path)
    """

    with open(file_path,'w') as file:
        file.write("ITEM: TIMESTEP \n")
        file.write(str(dump_class.timestep))
        file.write("\n")
        file.write("ITEM: NUMBER OF ATOMS \n")
        file.write(str(dump_class.numberofatoms))
        file.write("\n")
        file.write("ITEM: BOX BOUNDS ")
        file.write(str(dump_class.boxbounds.iloc[0,0]+" "+dump_class.boxbounds.iloc[0,1]+" "+dump_class.boxbounds.iloc[0,2]))
        bounds = pd.DataFrame(dump_class.boxbounds.loc["low"])
        bounds = bounds.join( pd.DataFrame(dump_class.boxbounds.loc["high"]))
        file.write(str(bounds.iloc[0:4,0:3]).replace("x  ","").replace("y  ","").replace("z  ","").replace("low ","").replace("high",""))
        file.write("\n")
        file.write("ITEM: ATOMS ")

    dump_class.atoms.to_csv(file_path,mode = "a", index = False,sep = ' ')

def file_class_to_lammps(data_class,file_path):
    """
    This writes a file() class to a file path

    file_class_to_lammps(data_class,file_path)
    """
    with open(file_path,'w') as file:
        file.write("# \n# ")
    data_class.data.to_csv(file_path, index = False,mode="a",sep = ' ')



def import_dumps_combined(files:list,removed_values:dict,filtered_values:dict):
    """
    this will import many lammps singular file dumps combing all of them into
    one pandas dataframe this is good for times that data is being trained on

    import_dumps_combined(files:list,removed_values:dict,filtered_values:dict)

    files = list of file names
    removed_values = Formatted Dictionary {"Column":[Value1,Value2,...]}
    filtered_values = Fromatted Dictionary {"Column":[low_value,high_value]}
    """

    for ind,file in enumerate(files):
        gc.collect()#clear what it can

        if ind == 0:
            #Setting base dataframe
            data = dumpFile(file).atoms

        else:
            #appending to base dataframe
            data_grab = dumpFile(file).atoms
            data = data.append(data_grab)
    print(data.head)
    #cleaning dataframe
    if filtered_values != {}:
        data = bf.filtering_data(data,filtered_values)

    if removed_values != {}:
        data = bf.removing_data(data,removed_values)

    return data
