from typing import Optional

from helper.protocolfedlearningclass import ProtocolFedLearning
from fedLearnLogic.Client import Client
from fedLearnLogic.AggregationServer import Aggregator

def fl_algorithm(fed_learning_class_instance: ProtocolFedLearning,
         inputfolder: Optional[str] = '/mnt/input',
         outputfolder: Optional[str] = '/mnt/output'):
    """
    A small linear regression example.
    """
    assert inputfolder is not None, "inputfolder must be provided"
    assert outputfolder is not None, "outputfolder must be provided"
    # CLIENT 1: get feature names
    client = Client(inputfolder=inputfolder)
    features = client.get_feature_names()
    fed_learning_class_instance.send_data_to_coordinator(data=features)

    # COORDINATOR 1: intersect feature
    if fed_learning_class_instance.is_coordinator:
        aggregator = Aggregator()
        features_list = fed_learning_class_instance.gather_data()
        common_features = aggregator.intersection_features(features_list)
        fed_learning_class_instance.broadcast_data(data=common_features)

    # CLIENT 2: calculate XtX and Xty
    common_features = fed_learning_class_instance.await_data()
    client.update_to_common_features(common_features)

    XtX = client.calculate_XtX()
    Xty = client.calculate_Xty()
    fed_learning_class_instance.send_data_to_coordinator(data=XtX, memo="XtX")
    fed_learning_class_instance.send_data_to_coordinator(data=Xty, memo="Xty")
        # fyi: the memo is actually optional and only is needed for when in one
        # aggregation round multiple different data is sent as it is the case here.
        # We also added the memo since we had race condition problems sometimes:
        # Sometimes for some algorithms a client managed to send data for a next round
        # before the first round was finished.
        # This messed up the first round
        # By default we have a counter and for every send_data_to_coordinator
        # we either use the given memo or a memo like "round1".
        # in the ROUND 1 before this this default memo is used.

    # COORDINATOR 2: calculate global beta and save model
    if fed_learning_class_instance.is_coordinator:
        XtX_list = fed_learning_class_instance.gather_data(memo="XtX")
        XtY_list = fed_learning_class_instance.gather_data(memo="Xty")
            # fyi, we use pickle for serialization, which is why we don't need to do any specific
            # serialization/deserialization for numpy here
        global_beta = aggregator.calculate_global_beta(XtX_list, XtY_list)
        fed_learning_class_instance.broadcast_data(data=global_beta, memo="global_beta")

    # CLIENT 3: await global beta and save model
    global_beta = fed_learning_class_instance.await_data(memo="global_beta")
    client.save_model(global_beta=global_beta, outputfolder=outputfolder)
