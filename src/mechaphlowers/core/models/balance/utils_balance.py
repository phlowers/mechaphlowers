import numpy as np



class MapVectorToNodeSpace:
    def __init__(self, nodes):
        self.nodes = nodes
        self.ntype = nodes.ntype
        self.tensor = self.init_tensor_space_matrix()
        self.tensor_flat = self.tensor.T.reshape(-1)
        self.register_nodes_values(nodes.Fx, nodes.Fz, nodes.My)

    def init_tensor_space_matrix(self):
        num = np.array(self.nodes.ntype, copy=True)
        num[-1] = 4
        num_1 = num * 10 + 1
        num_2 = num * 10 + 2
        num_3 = num * 10 + 3

        cat_num = np.array([num_1, num_2, num_3], copy=True)

        # TODO: reshape before filtering
        self.mask_tensor_flat_to_vector = np.isin(
            cat_num.T.reshape(-1), [33, 11, 12, 23, 43]
        )
        self.mask_vector_to_nodes = cat_num.T.reshape(-1)[
            self.mask_tensor_flat_to_vector
        ]

        self.mask_x = np.isin(self.mask_vector_to_nodes, [33, 11, 23, 43])

        self.mask_z = np.isin(self.mask_vector_to_nodes, [33, 12, 23, 43])

        return cat_num

    def filter_to_nodes_values(self, vector, ntype=1):
        return np.where(
            (self.mask_vector_to_nodes > ntype * 10)
            & (self.mask_vector_to_nodes < ntype * 10 + 9),
            vector,
            0,
        )

    def filter_vector(self, vector, ntype=1):
        if len(vector) != len(self.mask_vector_to_nodes):
            print("Error: vector length is not equal to mask")
        return np.where(
            self.mask_vector_to_nodes == ntype,
            vector,
            0,
        )

    def filter_coord(self, vector, coord="x"):
        if coord == "x":
            return vector[self.mask_x]
        # if coord == "y":
        #     return vector[self.mask_]
        if coord == "z":
            return vector[self.mask_z]

    def register_nodes_values(self, f_1, f_2, f_3):
        self.f_matrix_tensor_space = np.array([f_1, f_2, f_3], copy=True)

    def vector_value(self):
        return self.f_matrix_tensor_space.T.reshape(-1)[
            self.mask_tensor_flat_to_vector
        ]



class VectorProjection:
    def __init__(self):
        pass

    def set_tensions(self, Th, Tv_d, Tv_g):
        self.Th = Th
        self.Tv_d = Tv_d
        self.Tv_g = Tv_g

    def set_angles(self, alpha, beta, line_angle):
        self.alpha = alpha
        self.beta = beta
        self.line_angle = line_angle

    def set_proj_angle(self, proj_angle):
        self.proj_angle = proj_angle

    def set_all(self, Th, Tv_d, Tv_g, alpha, beta, line_angle, proj_angle):
        self.set_tensions(Th, Tv_d, Tv_g)
        self.set_angles(alpha, beta, line_angle)
        self.set_proj_angle(proj_angle)

    # properties?
    def T_attachments_plane_left(self):
        beta = self.beta
        Th = self.Th
        Tv_g = self.Tv_g
        alpha = self.alpha
        vg = Tv_g * np.cos(beta) - Th * np.sin(beta) * np.sin(alpha)
        hg = Tv_g * np.sin(beta) + Th * np.cos(beta) * np.sin(alpha)
        lg = Th * np.cos(alpha)
        # order x, y, z ?
        return np.array([lg, hg, vg])

    def T_attachments_plane_right(self):
        beta = self.beta
        Th = self.Th
        Tv_d = self.Tv_d
        alpha = self.alpha
        vd = Tv_d * np.cos(beta) + Th * np.sin(beta) * np.sin(alpha)
        hd = Tv_d * np.sin(beta) - Th * np.cos(beta) * np.sin(alpha)
        ld = -Th * np.cos(alpha)
        # order x, y, z ?
        return np.array([ld, hd, vd])

    def T_line_plane_left(self):
        lg, hg, vg = self.T_attachments_plane_left()
        proj_angle = self.proj_angle
        r_s_g = lg * np.cos(proj_angle) - hg * np.sin(proj_angle)
        r_t_g = lg * np.sin(proj_angle) + hg * np.cos(proj_angle)
        r_z_g = vg
        # order between s and t?
        return np.array([r_s_g, r_t_g, r_z_g])

    def T_line_plane_right(self):
        ld, hd, vd = self.T_attachments_plane_right()
        proj_angle = self.proj_angle
        r_s_d = ld * np.cos(proj_angle) - hd * np.sin(proj_angle)
        r_t_d = ld * np.sin(proj_angle) + hd * np.cos(proj_angle)
        r_z_d = vd
        # order between s and t?
        return np.array([r_s_d, r_t_d, r_z_d])
