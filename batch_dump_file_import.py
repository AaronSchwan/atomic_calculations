def bacth_file_dump_import(files,ids):

    def main():
        #import function called by concurent futures
        def pulling(file_path,id):
            dump_class = dumpFile.lammps_dump(file_path)
            return dump_class



        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(pulling,batch_file_paths)

            for ind,dump_class in enumerate(results):

                if batch_ids[ind] == "TimestepDefault":
                    id = dump_class.timestep
                else:
                    id = batch_ids[ind]

                dump_files[id] = dump_class

        return dump_files

    if __name__ == '__main__':
        main()
