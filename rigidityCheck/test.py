 # -------------------- generate vertices of elements --------------------#
        vertex_list = []
        elements_dict = {}
        for index in assembled:
            element = element_object_list[index]
            
            vertex_1 = AlgebraicChecker.CreateVertex(vertex_list, element.vertices[0].tolist())
            vertex_2 = AlgebraicChecker.CreateVertex(vertex_list, element.vertices[1].tolist())
            elements_dict[index] = [vertex_1, vertex_2]

        # -------------------- generate couplers --------------------#
        couplers = ElementObject.GetCouplers(assembled, element_object_list)
        couplers_dict = {}
        
        
        for coupler in couplers:
            point_1, point_2 = closest_points_between_segments(
                element_object_list[coupler[0]].vertices, element_object_list[coupler[1]].vertices
            )
            vertex_1 = AlgebraicChecker.CreateVertex(vertex_list, point_1, element_index=coupler[0])
            vertex_2 = AlgebraicChecker.CreateVertex(vertex_list, point_2, element_index=coupler[1])
            couplers_dict[coupler] = [vertex_1, vertex_2]

        vertex_num = len(vertex_list)

        # **************************************************************************
        # Step 1: generate constraints of constant length on segments including couplers
        # **************************************************************************

        const_length_constrains_vertex = []  # [[vertex_i, vertex_j]]

        for index in assembled:
            vertex_1 = elements_dict[index][0]

            # create constraints: segment
            vertex_2 = elements_dict[index][1]
            const_length_constrains_vertex.append([vertex_1, vertex_2])

            # create constraints: couplers
            for coupler, coupler_vertices in couplers_dict.items():
                coupler: Tuple
                if index in coupler:
                    vertex_2 = coupler_vertices[1 - coupler.index(index)]
                    const_length_constrains_vertex.append([vertex_1, vertex_2])

        for coupler in couplers:
            vertex_1, vertex_2 = couplers_dict[coupler]
            const_length_constrains_vertex.append([vertex_1, vertex_2])

        K_const_length = AlgebraicChecker.CreateConstLengthConstrains(const_length_constrains_vertex, vertex_num)

        # **************************************************************************
        # Step 2: generate constraints of rotation for couplers
        # **************************************************************************

        rotation_constrains_vertex = []  # [[vertex_i, vertex_j, vertex_k]]

        for coupler in couplers:
            coupler_vertex_1, coupler_vertex_2 = couplers_dict[coupler]
            coupler_vertex_1: Vertex
            coupler_vertex_2: Vertex

            if coupler_vertex_1.element_index in elements_dict.keys():
                segment_vertex = elements_dict[coupler_vertex_1.element_index][0]
                mid_vertex = coupler_vertex_1
                end_vertex = coupler_vertex_2
                rotation_constrains_vertex.append([segment_vertex, mid_vertex, end_vertex])

            if coupler_vertex_2.element_index in elements_dict.keys():
                segment_vertex = elements_dict[coupler_vertex_2.element_index][0]
                mid_vertex = coupler_vertex_2
                end_vertex = coupler_vertex_1
                rotation_constrains_vertex.append([segment_vertex, mid_vertex, end_vertex])

        K_rotation = AlgebraicChecker.CreateRotationConstrains(rotation_constrains_vertex, vertex_num)

        # **************************************************************************
        # Step 3: generate constraints of collinear for segments
        # **************************************************************************

        collinear_constrains_vertex = []  # [[vertex_i, vertex_j, vertex_k]]

        for coupler in couplers:
            coupler_vertex_1, coupler_vertex_2 = couplers_dict[coupler]
            coupler_vertex_1: Vertex
            coupler_vertex_2: Vertex

            if coupler_vertex_1.element_index in elements_dict.keys():
                vertex_start: Vertex = elements_dict[coupler_vertex_1.element_index][0]
                vertex_end: Vertex = elements_dict[coupler_vertex_1.element_index][1]
                vertex_mid = coupler_vertex_1
                collinear_constrains_vertex.append([vertex_start, vertex_mid, vertex_end])

            if coupler_vertex_2.element_index in elements_dict.keys():
                vertex_start: Vertex = elements_dict[coupler_vertex_2.element_index][0]
                vertex_end: Vertex = elements_dict[coupler_vertex_2.element_index][1]
                vertex_mid = coupler_vertex_2
                collinear_constrains_vertex.append([vertex_start, vertex_mid, vertex_end])

        K_collinear = AlgebraicChecker.CreateCollinearConstrains(collinear_constrains_vertex, vertex_num)

        # **************************************************************************
        # Step 4: generate constraints of grounded segments
        # **************************************************************************

        grounded_constrains_vertex = []  # [vertex_i]

        for index in assembled:
            element = element_object_list[index]
            if element.is_grounded:
                grounded_constrains_vertex.append(elements_dict[index][0])
                grounded_constrains_vertex.append(elements_dict[index][1])

        K_grounded = AlgebraicChecker.CreateGroundedConstrains(grounded_constrains_vertex, vertex_num)

        K = K_const_length
        if K_rotation is not None:
            K = np.vstack((K, K_rotation))
        if K_collinear is not None:
            K = np.vstack((K, K_collinear))
        if K_grounded is not None:
            K = np.vstack((K, K_grounded))

        if np.linalg.matrix_rank(K) == vertex_num * 3:
            return ElementStatus.fixed