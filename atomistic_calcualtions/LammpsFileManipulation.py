"""
This is a set of useful features for manipulating basic lammps output files

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
import warnings
import concurrent.futures


#non-default imports
import pandas as pd


################################################################################
#Dealing with lammps dump files#################################################
################################################################################
class dumpFile:

    """
    This will read a file path that is either a pickle file or a lammps dump
    file

    obj = dumpFile(timestep:int,numberofatoms:int,boxbounds:pd.DataFrame,atoms:pd.DataFrame,serial=None)

    valid calls:
    obj.timestep = returns timestep in the file[int]
    obj.numberofatoms = numbers of atoms in the dump[int]
    obj.boxbounds = returns the bounds with type,low,high in a pandas datframe[pd.DataFrame]
    obj.atoms = atomic data[pd.DataFrame]
    obj.serial = serial number assigned to function[unassigned]

    class methods:

    dumpFile.lammps_dump(self, file_path) ##Must be a singular timestep dumpFile
    dumpFile.bianary_dump(self, file_path) ##Must be a bianary of this class format

    file_path = path to file


    """
    class_tolerance = 12 #the accuarcy of the classes operational functions
    checking_tolerance = 5 #how many decimals the classes attributes will be checked to

    def __init__(self,timestep:int,boundingtypes:list,atoms:pd.DataFrame):
        self.timestep = timestep
        self.boundingtypes = boundingtypes
        self.atoms = atoms

    #property defined functions#################################################
    @property
    def numberofatoms(self):
        #finds the number of atoms for the given data
        return len(self.atoms["id"])

    @property
    def boxbounds(self):

        #getting bounds and making custom format to values
        box_dict = {"labels":["type","low","high"],"x":[self.boundingtypes[0],min(self.atoms["x"]),max(self.atoms["x"])],"y":[self.boundingtypes[1],min(self.atoms["y"]),max(self.atoms["y"])],"z":[self.boundingtypes[2],min(self.atoms["z"]),max(self.atoms["z"])]}
        return pd.DataFrame.from_dict(box_dict).set_index("labels")

    @property
    def volume(self):
        """
        gets the volume of the overall simulation cell as a box
        """
        x_range = abs(self.boxbounds.loc["high","x"] - self.boxbounds.loc["low","x"])
        y_range = abs(self.boxbounds.loc["high","y"] - self.boxbounds.loc["low","y"])
        z_range = abs(self.boxbounds.loc["high","z"] - self.boxbounds.loc["low","z"])

        return x_range*y_range*z_range

    @property
    def xy_area(self):
        """
        gets the area of the xy plane of the simulation cell as a box
        """
        x_range = abs(self.boxbounds.loc["high","x"] - self.boxbounds.loc["low","x"])
        y_range = abs(self.boxbounds.loc["high","y"] - self.boxbounds.loc["low","y"])

        return x_range*y_range

    @property
    def xz_area(self):
        """
        gets the area of the xy plane of the simulation cell as a box
        """
        x_range = abs(self.boxbounds.loc["high","x"] - self.boxbounds.loc["low","x"])
        z_range = abs(self.boxbounds.loc["high","z"] - self.boxbounds.loc["low","z"])

        return x_range*z_range

    @property
    def yz_area(self):
        """
        gets the area of the xy plane of the simulation cell as a box
        """
        z_range = abs(self.boxbounds.loc["high","z"] - self.boxbounds.loc["low","z"])
        y_range = abs(self.boxbounds.loc["high","y"] - self.boxbounds.loc["low","y"])

        return z_range*y_range


    #Alternative CLass Constructive Methods#####################################
    @classmethod
    def lammps_dump(cls,file_path:str):
        """
        uses path of raw lammps file **Must be a singular timestep

        will create the class of dumpFile once processed
        """
        raw_data = pd.read_csv(file_path)#getting data

        titles = raw_data.iloc[7].str.split(expand = True).iloc[0][2:].tolist()#getting titles of atomic data
        data = pd.DataFrame(raw_data.iloc[8:,0].str.split(' ',len(titles)-1).tolist(), columns = titles)#getting atomic data
        data =  data.apply(pd.to_numeric)

        #getting bounds and making custom format to values
        boxboundtype =  raw_data.iloc[3,0].replace("ITEM: BOX BOUNDS ","").split(" ")

        #returning class
        return cls(int(raw_data.iloc[0,0]),boxboundtype,data)

    #IMPORTANT redefine this to work with LAMMPS Dummps bianary
    @classmethod
    def bianary_dump(cls,file_path:str):
        """
        creates class from a bianary dumpFile class
        """
        #opening class saved as bianary
        with open(file_path, 'rb') as input:
            data = pickle.load(input)
        #returning class
        return cls(data.timestep,data.numberofatoms,data.boxbounds,data.atoms)

    @classmethod
    def raw_file_data(cls,raw_data:pd.DataFrame):
        """
        takes in a lammps dump file in the form of a singular column singular time step

        **Must include all data from first row "TimeStep:" to last
        """

        titles = raw_data.iloc[8].str.split(expand = True).iloc[0][2:].tolist()#getting titles of atomic data
        data = pd.DataFrame(raw_data.iloc[9:,0].str.split(' ',len(titles)-1).tolist(), columns = titles)#getting atomic data
        data =  data.apply(pd.to_numeric)

        #getting bounds and making custom format to values
        boxboundtype =  raw_data.iloc[4,0].replace("ITEM: BOX BOUNDS ","").split(" ")
        boxboundings = pd.DataFrame(raw_data.iloc[5:8,0].str.split(' ',1).tolist())
        box_dict = {"labels":["type","low","high"],"x":[boxboundtype[0],boxboundings.iloc[0,0],boxboundings.iloc[0,1]],"y":[boxboundtype[1],boxboundings.iloc[1,0],boxboundings.iloc[1,1]],"z":[boxboundtype[2],boxboundings.iloc[2,0],boxboundings.iloc[2,1]]}

        #returning class
        return cls(int(raw_data.iloc[1,0]),int(raw_data.iloc[3,0]),pd.DataFrame.from_dict(box_dict).set_index("labels"),data)

    #Class methods##############################################################
    @classmethod
    def change_checking_tolerance(cls,value):
        cls.checking_tolerance = value

    @classmethod
    def change_class_tolerance(cls,value):
        cls.class_tolerance = value

    #dubble under functions#####################################################
    def __repr__(self):
        #returning the atoms by default when calling the function alone
        return "{TimeStep:"+str(self.timestep)+"\nBoundings"+str(self.boxbounds)+"\nAtomic Data"+str(self.atoms.head)+"}"

    def __eq__(self):
        """
        This checks if the base conditions are equal such as the number of atoms,
        the boundary after transform to the first boundary, then if it passes
        both of those conditions it will check the atoms positions within a given
        tolerance(class variable name = class_tolerance)
        """
        pass;
    def __add__(self,other):
        """
        This is an alternative merge method first the class checks the compatability
        of the merge
        """
        pass;

    def __sub__(self,other):
        """
        This is an alternative method to remove columns from the
        """
        pass;


def multiple_timestep_singular_file_dumps(file_path:str,ids:list = ["TimestepDefault"]):
    """
    this opens a multi-timestep lammps dump and converts it to a dictionary of
    dumpFile classes with the keys set to the timesteps

    ids:list = ["TimestepDefault"]
    ids are set to the dumpclass timestep by default however if there are duplicates
    this will override the timesteps so you can define the ids for the dictionary
    """
    dump_files = {} #dictionary of class

    raw_data = pd.read_csv(file_path,header = None)#getting data
    indexes = raw_data.index[raw_data[0] == "ITEM: TIMESTEP"].tolist()#getting splitting indexes

    if len(ids) == len(indexes) or ids == ["TimestepDefault"]:

        #splitting and iterating through pandas dataFrame
        for ind, index in enumerate(indexes):

            if ind < len(indexes)-1:
                #all except last index
                df = raw_data.loc[index:indexes[ind+1]-1,:]#making new dataFrame
                dump_class = dumpFile.raw_file_data(df)#dump class processing

            else:
                #last index
                df = raw_data.loc[index:len(raw_data)+1,:]#making new dataFrame
                dump_class = dumpFile.raw_file_data(df)#dump class processing

            #adding to dictionary
            if ids == ["TimestepDefault"]:
                #using timestep to insert
                dump_files[int(dump_class.timestep)] = dump_class
            else:
                #using custom id
                dump_files[ids[ind]] = dump_class

        return dump_files


    else:
         warnings.warn("Length of ids list is not equal to files list length")

#def batch_import_files(file_paths:list,ids:list = ["TimestepDefault"], max_simultanius_processes:int = 5):
"""
    batch_import_files(file_paths:list,ids:list = ["TimestepDefault"], max_simultanius_processes:int = 5):

    ids:list = ["TimestepDefault"]
    ids are set to the dumpclass timestep by default however if there are duplicates
    this will override the timesteps so you can define the ids for the dictionary

    This will import several files into a dictionary of dumpFiles
    dict[id] = dumpFile

    this is a predone way to use the concurent.futures module to import several files
    accross several cores at once

    default processes = 5
"""
"""
    #adjusting ids if necessary
    if ids == ["TimestepDefault"]:
        ids = ["TimestepDefault" for i in range(len(file_paths))]

    if len(ids) == len(file_paths):




        def chunking(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]



        dump_files = {}


        #batching ids and files
        batching_file_paths = [file_paths[i:i + max_simultanius_processes] for i in range(0, len(file_paths), max_simultanius_processes)]
        batching_ids = [ids[i:i + max_simultanius_processes] for i in range(0, len(ids), max_simultanius_processes)]




    else:
         warnings.warn("Length of ids list is not equal to files list length")

"""

def write_lammps_dump(file_path:str,dump_class:dumpFile,mode:str = "a"):
    """
    This takes in a file path and writes a dumpFile class to the file path in
    standard lammps format

    write_lammps_dump(file_path:str,dump_class:dumpFile,mode:str = "a")

    file_path = path to file [str]
    dump_class = dumpFile class to be written [dumpFile]
    mode = overwrite("w") or append("a") **default append [str]

    """

    with open(file_path,mode) as file:
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

def write_dump_to_data_format(dump_class:dumpFile,file_path:str):
    """
    writes dumpFile class to a data file format

    **this will only save the positions in data fromatting
    **primary use to write a initiallization data file for a lammps

    ** example

    # LAMMPS data file written by LammpsFileManipulation.py
    275184 atoms
    2 atom types
    0.4892064609 119.789657019 xlo xhi
    -158.7268972078 158.7268972078 ylo yhi
    0.4917072972 119.7871561826 zlo zhi

    Atoms  # atomic

    1 2 2.59911 -158.671 2.89486
    .
    .
    .

    """

    with open(file_path,"w") as file:
        file.write("# LAMMPS data file written by LammpsFileManipulation.py \n")
        file.write(str(dump_class.numberofatoms))
        file.write(" atoms \n")
        file.write(str(max(dump_class.atoms["type"])))
        file.write("atom types \n")
        file.write(str("{:.15f}".format(float(test.boxbounds["x"].loc["low"]))+" "+"{:.15f}".format(float(test.boxbounds["x"].loc["high"]))+" xlo xhi"))
        file.write(str("{:.15f}".format(float(test.boxbounds["y"].loc["low"]))+" "+"{:.15f}".format(float(test.boxbounds["y"].loc["high"]))+" ylo yhi"))
        file.write(str("{:.15f}".format(float(test.boxbounds["z"].loc["low"]))+" "+"{:.15f}".format(float(test.boxbounds["z"].loc["high"]))+" zlo zhi"))
        file.write("\n\n")
        file.write("Atoms  # atomic\n\n")

    atomic_data = dump_class.atoms[["id","type", "x", "y", "z"]].to_csv(file_path,mode = "a", index = False,header = Flase ,sep = ' ')




dump_class = dumpFile.lammps_dump(r"C:\Users\Aaron Schwan\Desktop\Moments.0001_dump.221.0")


################################################################################
#Dealing with lammps data files#################################################
################################################################################

"""

class dataFile:

    def __init__(self,atoms,bonds,angles,dihedrals,dihedrals,impropers,atom_types,bond_types,angle_types,dihedral_types,improper_types,extra_bond_per_atom,ellipsoids,lines,triangles,xlo_xhi,ylo_yhi,zlo_zhi,xy_xz_yz):



        ++
        dihedrals = # of dihedrals in system
        impropers = # of impropers in system
        atom_types = # of atom types in system
        bond_types = # of bond types in system
        angle_types = # of angle types in system
        dihedral_types = # of dihedral types in system
        improper_types = # of improper types in system
        extra_bond_per_atom = leave space for this many new bonds per atom
        ellipsoids = # of ellipsoids in system
        lines = # of line segments in system
        triangles = # of triangles in system
        xlo_xhi = simulation box boundaries in x dimension
        ylo_yhi = simulation box boundaries in y dimension
        zlo_zhi = simulation box boundaries in z dimension
        xy_xz_yz = simulation box tilt factors for triclinic system

"""
