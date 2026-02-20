import tqdm
import math
import sys
import mosek
import numpy as np
import matplotlib.pyplot as plt


print_detail = True
#kn = 5.6e6


class ContFace():
    """Contact face class
    """

    def __init__(self, id, height, fc, ft=0):
        """
        :param id: ID of the face
        :type id: int
        :param height: Height of the face
        :type height: float
        :param fc: Compressive strength
        :type fc: float
        """
        self.id = id
        self.contps = []
        self.height = height
        self.fc = fc
        self.ft = ft

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False


def compute_thickness(contps, elems):
    thickness_dict = {}
    for p in contps.values():
        # element_vertices = elems[p.cand].vertices
        # #create polygon from vertices
        # polygon = Polygon(element_vertices)
        # #create line from contact point normal
        # line = LineString([p.coor, (p.coor[0]+p.normal[0],p.coor[1]+p.normal[1])])
        # #calculate intersection
        # intersection = polygon.intersection(line)
        # #calculate thickness
        # thickness = intersection.length
        # if thickness == 0:
        #     thickness = 1e-10
        # thickness_dict[p.id] = thickness

        vector_to_element_center = np.asarray(elems[p.cand].center)-np.asarray(p.coor)
        #project displacement to normal direction
        reversed_normal = -1*np.asarray(p.normal)
        normal_disp = np.dot(vector_to_element_center,reversed_normal)
        thickness = abs(normal_disp)
        thickness_dict[p.id] = thickness
        #print(thickness)
    # plt.hist(list(thickness_dict.values()),bins=100)
    # plt.show()
    return thickness_dict


def solve_elastic_finitefc_associative_3d_disp_control(elems, contps, Aglobal=None,thickness_dict=None,\
                                                       control_element=None,control_dof=1,\
                                                        control_displacement = 1,BC='cantilever',control_dof_beam = []):

    # result container
    result = dict()
    # # assemble contact faces
    # contfs = dict()
    # for p in tqdm.tqdm(contps.values(), desc='assemble contact faces'):
    #     if p.faceID not in contfs.keys():
    #         face = ContFace(p.faceID, p.section_h,
    #                         p.cont_type.fc, p.cont_type.ft)
    #         contfs[face.id] = face
    #         contfs[p.faceID].contps.append(p.id)
    #     else:
    #         contfs[p.faceID].contps.append(p.id)

    # nb_contfs = len(contfs)
    inf = 0.0

    def streamprinter(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    limit_force = 0
    # Make mosek environment
    with mosek.Env() as env:
        # Create a task object
        with env.Task(0, 0) as task:
            # Attach a log stream printer to the task
            if print_detail:
                task.set_Stream(mosek.streamtype.log, streamprinter)

           # Bound keys and values for constraints
            bkc = []
            blc = []
            buc = []
            for element in elems.values():
                #print(element.type)
                if element.type == "ground":
                    bkc.extend([mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr,
                                mosek.boundkey.fr])
                    blc.extend([-inf, -inf, -inf, -inf, -inf, -inf])
                    buc.extend([inf, inf, inf, inf, inf, inf])
                # elif element.type == "beam" and BC=='db':
                #     bkc.extend([mosek.boundkey.fx,
                #             mosek.boundkey.fx,
                #             mosek.boundkey.fx,
                #             mosek.boundkey.fr,
                #             mosek.boundkey.fr,
                #             mosek.boundkey.fr])
                #     blc.extend([element.dl[0], element.dl[1], element.dl[2], -inf, -inf, -inf])
                #     buc.extend([element.dl[0], element.dl[1], element.dl[2], inf, inf, inf])
                elif element.type == "beam":
                    # for i in range(6):
                    #     if i in control_dof_beam:
                    #         bkc.append(mosek.boundkey.fx)
                    #         blc.append(element.dl[i])
                    #         buc.append(element.dl[i])
                    #     else:
                    #         bkc.append(mosek.boundkey.fr)
                    #         blc.append(-inf)
                    #         buc.append(inf)
                    bkc.extend([mosek.boundkey.fx,
                            mosek.boundkey.fx,
                            mosek.boundkey.fx,
                            mosek.boundkey.fx,
                            mosek.boundkey.fx,
                            mosek.boundkey.fx])
                    blc.extend([element.dl[0], element.dl[1], element.dl[2], -inf, -inf, -inf])
                    buc.extend([element.dl[0], element.dl[1], element.dl[2], inf, inf, inf])
                else:
                    bkc.extend([mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx,
                                mosek.boundkey.fx])
                    blc.extend([element.dl[0],
                                element.dl[1], element.dl[2], element.dl[3],
                                element.dl[4], element.dl[5]])
                    buc.extend([element.dl[0],
                                element.dl[1], element.dl[2], element.dl[3],
                                element.dl[4], element.dl[5]])
            # #restrain the translation in y and rotation in x and rotatio in z
            # bkc_counter=0
            # #print(len(bkc)-6-2)
            # for bkc_counter in range(0,len(bkc)):
            #     if bkc_counter % 6 == 1 or bkc_counter % 6 == 3 or bkc_counter % 6 == 5:
            #         bkc[bkc_counter] = mosek.boundkey.fr
            #         blc[bkc_counter] = -inf
            #         buc[bkc_counter] = +inf
            #     # if bkc_counter==3*6+4 or bkc_counter==6*6+4 or bkc_counter==10*6+4 or \
            #     #     bkc_counter==len(bkc)-6-2:
            #     #     bkc[bkc_counter] = mosek.boundkey.fr
            #     #     blc[bkc_counter] = -inf
            #     #     buc[bkc_counter] = +inf
            for point in contps.values():  # 4th variable
                if (elems[point.anta].type.startswith('stone') and elems[point.cand].type.startswith('mortar'))\
                    or (elems[point.anta].type.startswith('mortar') and elems[point.cand].type.startswith('stone')):
                    factor = 1*point.section_h
                elif elems[point.anta].type.startswith('mortar') and elems[point.cand].type.startswith('mortar'):
                    factor = 1*point.section_h
                elif elems[point.anta].type.startswith('stone') and elems[point.cand].type.startswith('stone'):
                    factor = 1*point.section_h
                else:
                    factor = 1*point.section_h
                bkc.extend([mosek.boundkey.fx])
                blc.extend([-point.cont_type.cohesion*factor])
                buc.extend([-point.cont_type.cohesion*factor])
            
            #
            for key, value in contps.items():
                for i in range(3):
                    bkc.extend([mosek.boundkey.fx])
                    blc.extend([0])
                    buc.extend([0])

            # Bound keys for variables
            bkx = []
            blx = []
            bux = []
            for key, value in contps.items():
                bkx.extend([mosek.boundkey.fr, mosek.boundkey.fr,mosek.boundkey.ra,\
                            mosek.boundkey.fr, mosek.boundkey.fr,mosek.boundkey.fr, mosek.boundkey.fr])
                # if elems[value.anta].type.startswith('stone') or elems[value.cand].type.startswith('stone'):
                #     factor = 0.5
                # else:
                #     factor = 1
                factor_fc = 1*value.section_h
                ft = -value.cont_type.ft*factor_fc
                fc = value.cont_type.fc*factor_fc
                blx.extend([-inf, -inf, ft,-inf, -inf,-inf, -inf])
                bux.extend([+inf, +inf, fc,+inf, +inf,+inf, +inf])
                # for i in range(4):
                #     bkx.append(mosek.boundkey.fr)
                #     blx.append(-inf)
                #     bux.append(+inf)

            bkx.append(mosek.boundkey.lo)
            blx.append(0)
            bux.append(+inf)
            # set numerically reasonable value for m
            max_gap = abs(control_displacement)
            for p in contps.values():
                for gap_i in range(3):
                    if abs(p.gap[gap_i]) > max_gap:
                        max_gap = abs(p.gap[gap_i])
            #print('max_gap:',max_gap)
            bkx.append(mosek.boundkey.fx)
            blx.append(max_gap)
            bux.append(max_gap)
            #apply displacement control
            for i in range(6):
                bkx.append(mosek.boundkey.fr)
                blx.append(-inf)
                bux.append(+inf)

           

            # Objective coefficients
            c = []
            for key, value in tqdm.tqdm(contps.items(), desc='Objective coefficients'):
                # for i in range(2):  # 2variables(t,n)*1nodes*2contact faces
                c.extend([value.gap[0], value.gap[1],value.gap[2],0,0,0,0])
                # print(-g[g_index])
            c.append(max_gap)
            c.append(0)
            #apply displacement control
            for i in range(6):
                if i==control_dof:
                    c.append(control_displacement)
                else:
                    c.append(0)

            # determine index of control element
            control_index = None
            for element_index, value in enumerate(elems.values()):
                if value.type == control_element:
                    control_index = element_index
                    break
            #a_vector_p_to_e_forces = np.zeros((1,3*len(contps)))

            # Below is the sparse representation of the A
            # matrix stored by column.
            asub = []
            aval = []
            for i, point in tqdm.tqdm(enumerate(contps.values()), desc='assemble A matrix in mosek',total=len(contps)):
                thickness = point.thickness
                # if elems[point.cand].type.startswith('stone'):
                #     E = 1.69e10*0.1
                #     lamda = 0.2
                # elif elems[point.cand].type.startswith('mortar'):
                #     E = 3.23e8*0.1
                #     lamda = 0.2
                # else:
                #     E = 1.69e10*0.1
                #     lamda = 0.2
                E = point.cont_type.E
                lamda = point.cont_type.lamda
                kjn = E/thickness#approximation
                #kjn = 1e-1
                kn = kjn*point.section_h
                #kn = 1e8
                kt = kn/(2*(1+lamda))

                for j in range(3):
                    col_A = i*3+j
                    col_index = []
                    col_value = []
                    if type(Aglobal) is tuple:  # sparse matrix
                        col_index.extend(Aglobal[0][col_A])
                        col_value.extend(Aglobal[1][col_A])
                        #a_vector_p_to_e_forces[0,col_A] = Aglobal[1][col_A][control_index*6+control_dof]
                    else:
                        for element_id in range(len(elems)):
                            for equ in range(6):
                                row_A = element_id*6+equ
                                row = element_id*6+equ
                                if Aglobal[row_A][col_A] != 0:
                                    col_index.append(row)
                                    col_value.append(Aglobal[row_A][col_A])
                    if j==0 or j==1:
                        col_index.append(len(elems)*6+len(contps)+3*i+j)
                        col_value.append(1/np.sqrt(kt))
                        asub.extend([col_index])
                        aval.extend([col_value])

                    if j == 2:  # add extra 4-8th variable for each contact point
                        # if point.cont_type.mu > 0:
                        # col_index.append(len(elems)*6+i)
                        # col_value.append(point.cont_type.mu)
                        # asub.extend([col_index, [len(elems)*6+i]])
                        # aval.extend([col_value, [-1]])

                        col_index.append(len(elems)*6+i)
                        col_value.append(point.cont_type.mu)
                        col_index.append(len(elems)*6+len(contps)+3*i+j)
                        col_value.append(1/np.sqrt(kn))
                        asub.extend([col_index, [len(elems)*6+i], [len(elems)*6+len(contps)+3*i+0], \
                                     [len(elems)*6+len(contps)+3*i+1], [len(elems)*6+len(contps)+3*i+2]])
                        aval.extend([col_value, [-1], [-1], [-1], [-1]])

                    

                    # else:
                    #     asub.append(col_index)
                    #     aval.append(col_value)

            col_index = []
            col_value = []
            # i = 0
            # for element in elems.values():
            #     col_index.extend([6*i, 6*i+1, 6*i+2, 6*i+3, 6*i+4, 6*i+5])
            #     col_value.extend(
            #         [-element.ll[0], -element.ll[1], -element.ll[2], -element.ll[3], -element.ll[4], -element.ll[5]])
            #     i += 1
            asub.append(col_index)
            aval.append(col_value)
            asub.append(col_index)
            aval.append(col_value)
            for i in range(6):
                if i==control_dof:
                    # element_index_in_Equilibrium = 0
                    # for value in elems.values():
                    #     if value.type == control_element:
                    col_index = [6*control_index+control_dof]
                    col_value = [1]
                    asub.append(col_index)
                    aval.append(col_value)
                        #     break
                        # element_index_in_Equilibrium += 1
                else:
                    asub.append([])
                    aval.append([])

                

            #------------------Append con in the objective function------------------
            numvar = len(bkx)
            numcon = len(bkc)
            task.appendcons(numcon)
            task.appendvars(numvar)

            for j in tqdm.tqdm(range(numvar), desc='input var data to mosek'):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])

                # Set the bounds on variable j
                # blx[j] <= x_j <= bux[j]
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                # Input column j of A
                task.putacol(j,                  # Variable (column) index.
                             # Row index of non-zeros in column j.
                             asub[j],
                             aval[j])            # Non-zero Values of column j.

            # Set the bounds on constraints.
             # blc[i] <= constraint_i <= buc[i]

            for i in tqdm.tqdm(range(numcon), desc='input con data to mosek'):
                task.putconbound(i, bkc[i], blc[i], buc[i])

            # # Set up and input quadratic objective
            # qsubi = []
            # qsubj = []
            # qval = []
            # cont_index = 0
            # for key, value in contps.items():
            #     thickness = thickness_dict[value.id]
            #     if elems[value.cand].type.startswith('stone') or elems[contps[value.counterPoint].cand].type.startswith('stone'):
            #         E = 310e5
            #         lamda = 0.2
            #     elif elems[value.cand].type.startswith('mortar') and elems[contps[value.counterPoint].cand].type.startswith('mortar'):
            #         E = 310e3
            #         lamda = 0.2
            #     else:
            #         E = 310e5
            #         lamda = 0.2
            #     kjn = E/thickness#approximation
            #     #kjn = 1e-1
            #     kn = kjn*value.section_h/3
            #     #kn = 1e8
            #     kt = kn/(2*(1+lamda))
               
            #     qsubi.extend([cont_index, cont_index+1,cont_index+2])
            #     qsubj.extend([cont_index, cont_index+1,cont_index+2])
            #     qval.extend([1/kt,1/kt, 1/kn])
            #     cont_index += 3
            # task.putqobj(qsubi, qsubj, qval)

            #-----------------------------------Put con in the contsraint function---------------------
            # define the optimization task
            v_indices = []
            for i in range(len(contps)):
                task.appendcone(mosek.conetype.quad,
                                0.0,
                                [7*i+3, 7*i+0, 7*i+1])
                v_indices.extend([7*i+4,7*i+5,7*i+6])
            start_qindex = len(contps)*7
            v_indices.insert(0,start_qindex)
            v_indices.insert(1,start_qindex+1)
            task.appendcone(mosek.conetype.rquad,
                            0.0,
                            v_indices)
            # # Set up and input quadratic objective
            # qsubi = []
            # qsubj = []
            # qval = []
            # cont_index = 0
            # for key, value in contps.items():               
            #     qsubi=[cont_index*4, cont_index*4+1,cont_index*4+3]
            #     qsubj=[cont_index*4, cont_index*4+1,cont_index*4+3]
            #     qval=[2*1,2*1, 2*1]
            #     task.putqconk(len(elems)*6+cont_index,qsubi, qsubj, qval)
            #     cont_index += 1
            #------------------------------------------------------------------

            # Input the objective sense (minimize/maximize)
            task.putobjsense(mosek.objsense.minimize)


            # # write the task to a file
            # task.writedata("/home/qiwang/Projects/28_RBSM_software/src/example.opf")

            task.optimize()
            if print_detail:
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.log)

            # Get status information about the solution
            #solsta_bas = task.getsolsta(mosek.soltype.bas)
            solsta_itr = task.getsolsta(mosek.soltype.itr)
            convergence = False
            unknown_status = False

            xx = [0.] * numvar
            y = [0.]*numcon

            #if(solsta_itr == mosek.solsta.optimal or solsta_itr==mosek.solsta.unknown):
            if(solsta_itr == mosek.solsta.optimal):
                task.getxx(mosek.soltype.itr,  # Request the interior-point solution.
                           xx)
                task.gety(mosek.soltype.itr, y)
                convergence = True
            elif (solsta_itr==mosek.solsta.unknown):
                if print_detail:
                    print("Unknown solution status")
                unknown_status = True
                task.getxx(mosek.soltype.itr,  # Request the interior-point solution.
                           xx)
                task.gety(mosek.soltype.itr, y)
                
            else:
                if print_detail:
                    print("Other solution status")


        result['convergence'] = convergence
        result['unknown_status'] = unknown_status
        result["contact_forces"] = xx[0:len(contps)*7]
        result["imposed_force"] = xx[len(xx)-6+control_dof]
        result["displacements"] = y[0:len(elems)*6]
        for y_index in range(len(result["displacements"])):
            result["displacements"][y_index] = result["displacements"][y_index]

    return result