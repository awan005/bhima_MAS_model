from python_packages.pynsim import Institution


class PuneInstitution(Institution):

    def update_node_properties(self, keyword, value):
        """
            Updating Node Properties
        """
        for node in self.nodes:
            node.keyword = value

    def query_df_to_update_properties(self, df, xcol, ycol, _property):
        tol = 0.00000000083  # hard coded tolerance 
        for node in self.nodes:
            for elem in df.iterrows():
                selected = elem[1]
                condition1 = abs(selected[xcol] - node.x)
                condition2 = abs(selected[ycol] - node.y)
                if (condition1 < tol) and (condition2 < tol):
                    node.__dict__.update({_property: selected[_property]})

    def query_df_to_update_properties_2(self, df, xcol, ycol, _property):
        i = 0
        j = len(self.nodes)
        for node in self.nodes:
            node.__dict__[_property] = float(df.loc[(round(df[xcol], 6) == round(node.X, 6)) &
                                                    (round(df[ycol], 6) == round(node.Y, 6)) &
                                                    (df["time"] == "1997-07-31")][_property])
            if i % 500 == 0:
                print(str(i) + " of " + str(j) + " agents updated")
            i += 1

    def query_df_to_update_hh_properties_3(self, df, xcol, ycol, _property):
        i = 0
        j = len(self.nodes)
        for node in self.nodes:
            node.__dict__[_property] = float(df.loc[(df[xcol] == node.X6) & (df[ycol] == node.Y6) &
                                                    (df["time"] == "1997-07-31")][_property])
            if i % 500 == 0:
                print(str(i) + " of " + str(j) + " agents updated")
            i += 1
