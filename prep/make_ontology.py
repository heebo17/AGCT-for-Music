import os


def make_ontology(file:str):
    fw=open(os.path.join(dir,"ontology.txt"),"w")
    with open (file) as fr:
        for line in fr:
            fields=line.split(";")
            subfields1=fields[1].split(",")
            subfields2=fields[2].split(",")
            subfields3=fields[3].split(",")
            len1=len(subfields1)
            len2=len(subfields2)
            len3=len(subfields3)
            name=fields[0]
            fw.write(name)
            fw.write("\t")
            fw.write("YLR331C")
            fw.write("\t")
            fw.write("Not_defined")
            fw.write("\t")
            fw.write("Saccharomyces cerevisiae")
            fw.write("\t")
            fw.write("13-Mar-09")
            fw.write("\t")
            fw.write("Exemplar sequence")
            fw.write("\t")
            fw.write("Saccharomyces Genome Database")
            fw.write("\t")
            fw.write("YLR331C questionalbe")
            fw.write("\t")
            fw.write("YLR331C MID2")
            fw.write("\t")
            fw.write("Not_defined")
            fw.write("\t")
            fw.write("JIP3")
            fw.write("\t")
            fw.write("Not_defined")
            fw.write("\t")
            fw.write("S000004326")
            fw.write("\t")
            fw.write("// ")
            for i in range(len1-1):
                fw.write(subfields1[i])
                fw.write(" ")
                fw.write(" /// ")
            fw.write(subfields1[len1-1])
            fw.write("\t")
            if fields[2]=="":
                fw.write("Not_defined")
                fw.write("\t")
            else:
                fw.write("// ")
                for i in range(len2-1):
                    fw.write(" ")
                    fw.write(subfields2[i])
                    fw.write(" /// ")
                fw.write(subfields2[len2-1])
            fw.write("\t")
            if fields[3]=="":
                fw.write("Not_defined")
                
            else:
                fw.write("// ")
                for i in range(len3-1):
                    fw.write(subfields3[i])
                    fw.write(" /// ")
                fw.write(subfields3[len3-1])
            fw.write("\n")

    fw.close()
    return



dir=os.path.join("prep","annotation")
make_ontology(os.path.join(dir,"ontology_9k.txt"))