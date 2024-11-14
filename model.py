import math

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.ops as ops
import numpy as np

from utils import compute_offsets, assign_label, generate_proposal
from loss import ClsScoreRegression, BboxRegression


class FeatureExtractor(nn.Module):
    """
    Image feature extraction with MobileNet.
    """
    def __init__(self, reshape_size=224, pooling=False, verbose=False):
        super().__init__()

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1]) # Remove the last classifier

        # average pooling
        if pooling:
            self.mobilenet.add_module('LastAvgPool', nn.AvgPool2d(math.ceil(reshape_size/32.))) # input: N x 1280 x 7 x 7

        for i in self.mobilenet.named_parameters():
            i[1].requires_grad = True # fine-tune all

    def forward(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape Nx3x224x224

        Outputs:
        - feat: Image feature, of shape Nx1280 (pooled) or Nx1280x7x7
        """
        num_img = img.shape[0]

        img_prepro = img

        feat = []
        process_batch = 500
        for b in range(math.ceil(num_img/process_batch)):
            feat.append(self.mobilenet(img_prepro[b*process_batch:(b+1)*process_batch]
                                    ).squeeze(-1).squeeze(-1)) # forward and squeeze
        feat = torch.cat(feat)

        if verbose:
            print('Output feature shape: ', feat.shape)

        return feat


class FastRCNN(nn.Module):
    def __init__(self, in_dim=1280, hidden_dim=256, num_classes=20, \
                roi_output_w=2, roi_output_h=2, drop_ratio=0.3):
        super().__init__()

        assert(num_classes != 0)
        self.num_classes = num_classes
        self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
        self.feat_extractor = FeatureExtractor()
        ##############################################################################
        # TODO: Declare the cls & bbox heads (in Fast R-CNN).                        #
        # The cls & bbox heads share a sequential module with a Linear layer,        #
        # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
        # Linear layer.                                                              #
        # The cls head is a Linear layer that predicts num_classes + 1 (background). #
        # The det head is a Linear layer that predicts offsets(dim=4).               #
        # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
        # hidden_dim -> hidden_dim.                                                  #
        ##############################################################################
        # Replace "pass" statement with your code
        self.shared_fc1 = nn.Linear(in_dim, hidden_dim)
        self.shared_drop = nn.Dropout(drop_ratio)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_fc = nn.Linear(hidden_dim, num_classes + 1)
        self.off_fc = nn.Linear(hidden_dim, 4)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, images, bboxes, bbox_batch_ids, proposals, proposal_batch_ids):
        """
        Training-time forward pass for our two-stage Faster R-CNN detector.

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - bboxes: Tensor of shape (N, 5) giving ground-truth bounding boxes
        and category labels, from the dataloader, where N is the total number
        of GT boxes in the batch
        - bbox_batch_ids: Tensor of shape (N, ) giving the index (in the batch)
        of the image that each GT box belongs to
        - proposals: Tensor of shape (M, 4) giving the proposals for input images, 
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image 
        that each proposals belongs to

        Outputs:
        - total_loss: Torch scalar giving the overall training loss.
        """
        w_cls = 1 # for cls_scores
        w_bbox = 1 # for offsets
        total_loss = None
        ##############################################################################
        # TODO: Implement the forward pass of Fast R-CNN.                            #
        # A few key steps are outlined as follows:                                   #
        # i) Extract image fearure.                                                  #
        # ii) Perform RoI Align on proposals, then meanpool the feature in the       #
        #     spatial dimension.                                                     #
        # iii) Pass the RoI feature through the shared-fc layer. Predict             #
        #      classification scores ans box offsets.                                #
        # iv) Assign the proposals with targets of each image.                       # 
        # v) Compute the cls_loss between the predicted class_prob and GT_class      #
        #    (For poistive & negative proposals)                                     #
        #    Compute the bbox_loss between the offsets and GT_offsets                #
        #    (For positive proposals)                                                #
        #    Compute the total_loss which is formulated as:                          #
        #    total_loss = w_cls*cls_loss + w_bbox*bbox_loss.                         #
        ##############################################################################
        # Replace "pass" statement with your code
        B, _, H, W = images.shape
        
        # extract image feature
        features = self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois = ops.roi_pool(features, torch.cat([proposal_batch_ids.unsqueeze(1), proposals], dim=1), output_size=(7, 7))
        pooled = nn.functional.adaptive_avg_pool2d(rois, (1, 1)).view(rois.size(0), -1)

        # forward heads, get predicted cls scores & offsets
        x = nn.functional.relu(self.shared_fc1(pooled))
        x = self.shared_drop(x)
        x = nn.functional.relu(self.shared_fc2(x))
        scores = self.cls_fc(x)
        offsets = self.off_fc(x)

        # assign targets with proposals
        pos_masks, neg_masks, GT_labels, GT_bboxes = [], [], [], []
        for img_idx in range(B):
            # get the positive/negative proposals and corresponding
            # GT box & class label of this image
            pos_mask, neg_mask, GT_label, GT_bbox = \
                assign_label(proposals[proposal_batch_ids == img_idx], \
                            bboxes[bbox_batch_ids == img_idx], self.num_classes)
            pos_masks.extend(pos_mask.tolist())
            neg_masks.extend(neg_mask.tolist())
            GT_labels.extend(GT_label.tolist())
            GT_bboxes.extend(GT_bbox.tolist())

        # compute loss
        GT_labels = torch.tensor(GT_labels, dtype=torch.long, device=scores.device)
        GT_bboxes = torch.tensor(GT_bboxes, dtype=torch.float32, device=proposals.device)
        cls_loss = ClsScoreRegression(scores[np.logical_or(pos_masks, neg_masks)], GT_labels[np.logical_or(pos_masks, neg_masks)], B)
        bbox_loss = BboxRegression(offsets[pos_masks], compute_offsets(proposals[pos_masks], GT_bboxes), B)
        total_loss = w_cls * cls_loss + w_bbox * bbox_loss
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return total_loss

    def inference(self, images, proposals, proposal_batch_ids, thresh=0.005, nms_thresh=0.007):
        """"
        Inference-time forward pass for our two-stage Faster R-CNN detector

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - proposals: Tensor of shape (M, 4) giving the proposals for input images, 
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image 
        that each proposals belongs to
        - thresh: Threshold value on confidence probability. HINT: You can convert the
        classification score to probability using a softmax nonlinearity.
        - nms_thresh: IoU threshold for NMS

        We can output a variable number of predicted boxes per input image.
        In particular we assume that the input images[i] gives rise to P_i final
        predicted boxes.

        Outputs:
        - final_proposals: List of length (B,) where final_proposals[i] is a Tensor
        of shape (P_i, 4) giving the coordinates of the final predicted boxes for
        the input images[i]
        - final_conf_probs: List of length (B,) where final_conf_probs[i] is a
        Tensor of shape (P_i, 1) giving the predicted probabilites that the boxes
        in final_proposals[i] are objects (vs background)
        - final_class: List of length (B,), where final_class[i] is an int64 Tensor
        of shape (P_i, 1) giving the predicted category labels for each box in
        final_proposals[i].
        """
        final_proposals, final_conf_probs, final_class = None, None, None
        ##############################################################################
        # TODO: Predicting the final proposal coordinates `final_proposals`,         #
        # confidence scores `final_conf_probs`, and the class index `final_class`.   #
        # The overall steps are similar to the forward pass, but now you cannot      #
        # decide the activated nor negative proposals without GT boxes.              #
        # You should apply post-processing (thresholding and NMS) to all proposals   #
        # and keep the final proposals.                                               #
        ##############################################################################
        # Replace "pass" statement with your code
        B = images.shape[0]

        # extract image feature
        features = self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois = ops.roi_pool(features, torch.cat([proposal_batch_ids.unsqueeze(1), proposals], dim=1), output_size=(7, 7))
        pooled = nn.functional.adaptive_avg_pool2d(rois, (1, 1)).view(rois.size(0), -1)

        # forward heads, get predicted cls scores & offsets
        x = nn.functional.relu(self.shared_fc1(pooled))
        x = self.shared_drop(x)
        x = nn.functional.relu(self.shared_fc2(x))
        scores = self.cls_fc(x)
        offsets = self.off_fc(x)

        # get predicted boxes & class label & confidence probability
        scores = scores[:, :-1]
        _, temp_class = scores.max(dim=1)
        temp_conf_probs = torch.take_along_dim(nn.functional.softmax(scores, dim=1), temp_class.unsqueeze(1), dim=1).squeeze(1)

        final_proposals = []
        final_conf_probs = []
        final_class = []
        # post-process to get final predictions
        for img_idx in range(B):
            # filter by threshold
            pos_mask = torch.logical_and(temp_conf_probs > thresh, proposal_batch_ids == img_idx)

            # nms
            p = proposals[pos_mask]
            cl = temp_class[pos_mask]
            conf = temp_conf_probs[pos_mask]
            p = generate_proposal(p, offsets[pos_mask])
            keep = ops.nms(p, conf, nms_thresh)

            # append to final predictions
            final_proposals.append(p[keep])
            final_conf_probs.append(conf[keep])
            final_class.append(cl[keep])

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return final_proposals, final_conf_probs, final_class