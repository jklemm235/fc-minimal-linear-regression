from numpy import linalg

class Aggregator:
    def __init__(self):
        pass

    def union_features(self, features_list):
        """Union the feature names from all clients."""
        all_features = set()
        for features in features_list:
            all_features.update(features)
        return list(all_features)

    def intersection_features(self, features_list):
        """Intersection of feature names from all clients."""
        if not features_list:
            return []
        common_features = set(features_list[0])
        for features in features_list[1:]:
            common_features.intersection_update(features)
        return list(common_features)

    def calculate_global_beta(self, XtX_list, Xty_list):
        """Calculate the global beta coefficients using the aggregated XtX and Xty."""
        # ensure that the dimensions of XtX and Xty are compatible
        if not XtX_list or not Xty_list:
            raise ValueError("XtX_list and Xty_list cannot be empty.")
        if len(XtX_list) != len(Xty_list):
            raise ValueError("XtX_list and Xty_list must have the same length.")
        for XtX, Xty in zip(XtX_list, Xty_list):
            if XtX.shape[0] != XtX.shape[1]:
                raise ValueError("Each XtX must be a square matrix.")
            if XtX.shape[1] != Xty.shape[0]:
                raise ValueError("The number of columns in XtX must match the number of rows in Xty.")
        # sum up the XtX and Xty from all clients
        global_XtX = sum(XtX_list)
        global_Xty = sum(Xty_list)
        # Solve for beta: beta = (X^T * X)^(-1) * (X^T * y)
        # pseudoinverse isnt the best solution but it is a solution
        if linalg.det(global_XtX) == 0:
            beta = linalg.pinv(global_XtX) @ global_Xty
        else:
            beta = linalg.inv(global_XtX) @ global_Xty
        return beta
