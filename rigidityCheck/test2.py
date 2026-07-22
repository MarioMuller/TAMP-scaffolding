    @staticmethod
    def BuildRigidityMatrix(
        assembled: List[int],
        element_object_list: List[ElementObject],
    ) -> RigidityMatrixResult:
        """
        Build the rigidity matrix for the assembled structure.

        Returns:
            matrix:
                Complete rigidity matrix K.

            vertex_list:
                Internal vertices used by the algebraic checker.

            elements_dict:
                Maps each element index to its two endpoint vertices.
        """

        # -------------------- generate vertices of elements --------------------#
        vertex_list = []
        elements_dict = {}

        for index in assembled:
            element = element_object_list[index]

            vertex_1 = AlgebraicChecker.CreateVertex(
                vertex_list,
                element.vertices[0].tolist(),
                element_index=index,
            )
            vertex_2 = AlgebraicChecker.CreateVertex(
                vertex_list,
                element.vertices[1].tolist(),
                element_index=index,
            )

            elements_dict[index] = [vertex_1, vertex_2]

        # -------------------- generate couplers --------------------#
        couplers = ElementObject.GetCouplers(
            assembled,
            element_object_list,
        )

        couplers_dict = {}

        for coupler in couplers:
            point_1, point_2 = closest_points_between_segments(
                element_object_list[coupler[0]].vertices,
                element_object_list[coupler[1]].vertices,
            )

            vertex_1 = AlgebraicChecker.CreateVertex(
                vertex_list,
                point_1,
                element_index=coupler[0],
            )
            vertex_2 = AlgebraicChecker.CreateVertex(
                vertex_list,
                point_2,
                element_index=coupler[1],
            )

            couplers_dict[coupler] = [vertex_1, vertex_2]

        vertex_num = len(vertex_list)

        # -------------------- constant-length constraints --------------------#
        const_length_constrains_vertex = []

        for index in assembled:
            vertex_1 = elements_dict[index][0]
            vertex_2 = elements_dict[index][1]

            const_length_constrains_vertex.append(
                [vertex_1, vertex_2]
            )

            for coupler, coupler_vertices in couplers_dict.items():
                if index in coupler:
                    vertex_2 = coupler_vertices[
                        1 - coupler.index(index)
                    ]
                    const_length_constrains_vertex.append(
                        [vertex_1, vertex_2]
                    )

        for coupler in couplers:
            vertex_1, vertex_2 = couplers_dict[coupler]
            const_length_constrains_vertex.append(
                [vertex_1, vertex_2]
            )

        K_const_length = AlgebraicChecker.CreateConstLengthConstrains(
            const_length_constrains_vertex,
            vertex_num,
        )

        # -------------------- rotation constraints --------------------#
        rotation_constrains_vertex = []

        for coupler in couplers:
            coupler_vertex_1, coupler_vertex_2 = couplers_dict[coupler]

            if coupler_vertex_1.element_index in elements_dict:
                segment_vertex = elements_dict[
                    coupler_vertex_1.element_index
                ][0]

                rotation_constrains_vertex.append(
                    [
                        segment_vertex,
                        coupler_vertex_1,
                        coupler_vertex_2,
                    ]
                )

            if coupler_vertex_2.element_index in elements_dict:
                segment_vertex = elements_dict[
                    coupler_vertex_2.element_index
                ][0]

                rotation_constrains_vertex.append(
                    [
                        segment_vertex,
                        coupler_vertex_2,
                        coupler_vertex_1,
                    ]
                )

        K_rotation = AlgebraicChecker.CreateRotationConstrains(
            rotation_constrains_vertex,
            vertex_num,
        )

        # -------------------- collinear constraints --------------------#
        collinear_constrains_vertex = []

        for coupler in couplers:
            coupler_vertex_1, coupler_vertex_2 = couplers_dict[coupler]

            if coupler_vertex_1.element_index in elements_dict:
                vertex_start = elements_dict[
                    coupler_vertex_1.element_index
                ][0]
                vertex_end = elements_dict[
                    coupler_vertex_1.element_index
                ][1]

                collinear_constrains_vertex.append(
                    [
                        vertex_start,
                        coupler_vertex_1,
                        vertex_end,
                    ]
                )

            if coupler_vertex_2.element_index in elements_dict:
                vertex_start = elements_dict[
                    coupler_vertex_2.element_index
                ][0]
                vertex_end = elements_dict[
                    coupler_vertex_2.element_index
                ][1]

                collinear_constrains_vertex.append(
                    [
                        vertex_start,
                        coupler_vertex_2,
                        vertex_end,
                    ]
                )

        K_collinear = AlgebraicChecker.CreateCollinearConstrains(
            collinear_constrains_vertex,
            vertex_num,
        )

        # -------------------- grounded constraints --------------------#
        grounded_constrains_vertex = []

        for index in assembled:
            element = element_object_list[index]

            if element.is_grounded:
                grounded_constrains_vertex.append(
                    elements_dict[index][0]
                )
                grounded_constrains_vertex.append(
                    elements_dict[index][1]
                )

        K_grounded = AlgebraicChecker.CreateGroundedConstrains(
            grounded_constrains_vertex,
            vertex_num,
        )

        # -------------------- combine matrices --------------------#
        K = K_const_length

        if K_rotation is not None:
            K = np.vstack((K, K_rotation))

        if K_collinear is not None:
            K = np.vstack((K, K_collinear))

        if K_grounded is not None:
            K = np.vstack((K, K_grounded))

        return RigidityMatrixResult(
            matrix=K,
            vertex_list=vertex_list,
            elements_dict=elements_dict,
        )

    @staticmethod
    def GetNullspaceModes(
        K: np.ndarray,
        tolerance: float | None = None,
    ) -> list[np.ndarray]:
        """
        Return infinitesimal displacement modes satisfying K @ mode ~= 0.
        """

        _, singular_values, vh = np.linalg.svd(
            K,
            full_matrices=True,
        )

        if tolerance is None:
            largest = singular_values[0] if len(singular_values) else 0.0
            tolerance = (
                max(K.shape)
                * np.finfo(float).eps
                * largest
            )

        rank = int(np.sum(singular_values > tolerance))

        return [
            mode.copy()
            for mode in vh[rank:]
        ]