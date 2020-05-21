import numpy as np
import torch
from manopth.manolayer import ManoLayer 

# FINGER LIMIT ANGLE FOR RIGHT HAND:
limit_bigfinger_right = torch.FloatTensor([1.2, -0.4, 0.25]) # 36:39
limit_index_right = torch.FloatTensor([-0.0827, -0.4389,  1.5193]) # 0:3
limit_middlefinger_right = torch.FloatTensor([-2.9802e-08, -7.4506e-09,  1.4932e+00]) # 9:12
limit_fourth_right = torch.FloatTensor([0.1505, 0.3769, 1.5090]) # 27:30
limit_small_right = torch.FloatTensor([-0.6235,  0.0275,  1.0519]) # 18:21

limit_secondjoint_bigfinger_right = torch.FloatTensor([0.0, -1.0, 0.0])
limit_secondjoint_index_right = torch.FloatTensor([0.0, 0.0, 1.2])
limit_secondjoint_middlefinger_right = torch.FloatTensor([0.0, 0.4, 1.2])
limit_secondjoint_fourth_right = torch.FloatTensor([0.0, 1.0, 1.0])
limit_secondjoint_small_right = torch.FloatTensor([0.0, 0.0, 1.2])

limit_thirdjoint_bigfinger_right = torch.FloatTensor([0.0, -1.0, 0.0])
limit_thirdjoint_index_right = torch.FloatTensor([0.0, 0.0, 1.2])
limit_thirdjoint_middlefinger_right = torch.FloatTensor([0.0, 0.4, 1.2])
limit_thirdjoint_fourth_right = torch.FloatTensor([0.0, 1.0, 1.0])
limit_thirdjoint_small_right = torch.FloatTensor([0.0, 0.0, 1.2])


# FINGER LIMIT ANGLE FOR LEFT HAND:
limit_bigfinger_left = torch.FloatTensor([1.2, -0.4, 0.25]) # 36:39
limit_index_left = torch.FloatTensor([0.0827, 0.4389,  -1.5193]) # 0:3
limit_middlefinger_left = torch.FloatTensor([2.9802e-08, 7.4506e-09,  -1.4932e+00]) # 9:12
limit_fourth_left = torch.FloatTensor([-0.1505, -0.3769, -1.5090]) # 27:30
limit_small_left = torch.FloatTensor([-0.6235, 0.1, -1.0519]) # 18:21

limit_secondjoint_bigfinger_left = torch.FloatTensor([0.0, 0.8, -0.8])
limit_secondjoint_index_left = torch.FloatTensor([0.0, 0.0, -1.2])
limit_secondjoint_middlefinger_left = torch.FloatTensor([0.0, -0.4, -1.2])
limit_secondjoint_fourth_left = torch.FloatTensor([0.0, -1.0, -1.0])
limit_secondjoint_small_left = torch.FloatTensor([0.0, 0.0, -1.0])

limit_thirdjoint_bigfinger_left = torch.FloatTensor([0.0, 0.8, -0.8])
limit_thirdjoint_index_left = torch.FloatTensor([0.0, 0.0, -1.2])
limit_thirdjoint_middlefinger_left = torch.FloatTensor([0.0, -0.4, -1.2])
limit_thirdjoint_fourth_left = torch.FloatTensor([0.0, -1.0, -1.0])
limit_thirdjoint_small_left = torch.FloatTensor([0.0, 0.0, -1.0])


if torch.cuda.is_available():
    limit_bigfinger_right = limit_bigfinger_right.cuda()
    limit_index_right = limit_index_right.cuda()
    limit_middlefinger_right = limit_middlefinger_right.cuda()
    limit_fourth_right = limit_fourth_right.cuda()
    limit_small_right = limit_small_right.cuda()

    limit_secondjoint_bigfinger_right = limit_secondjoint_bigfinger_right.cuda()
    limit_secondjoint_index_right = limit_secondjoint_index_right.cuda()
    limit_secondjoint_middlefinger_right = limit_secondjoint_middlefinger_right.cuda()
    limit_secondjoint_fourth_right = limit_secondjoint_fourth_right.cuda()
    limit_secondjoint_small_right = limit_secondjoint_small_right.cuda()

    limit_thirdjoint_bigfinger_right = limit_thirdjoint_bigfinger_right.cuda()
    limit_thirdjoint_index_right = limit_thirdjoint_index_right.cuda()
    limit_thirdjoint_middlefinger_right = limit_thirdjoint_middlefinger_right.cuda()
    limit_thirdjoint_fourth_right = limit_thirdjoint_fourth_right.cuda()
    limit_thirdjoint_small_right = limit_thirdjoint_small_right.cuda()

    limit_bigfinger_left = limit_bigfinger_left.cuda()
    limit_index_left = limit_index_left.cuda()
    limit_middlefinger_left = limit_middlefinger_left.cuda()
    limit_fourth_left = limit_fourth_left.cuda()
    limit_small_left = limit_small_left.cuda()

    limit_secondjoint_bigfinger_left = limit_secondjoint_bigfinger_left.cuda()
    limit_secondjoint_index_left = limit_secondjoint_index_left.cuda()
    limit_secondjoint_middlefinger_left = limit_secondjoint_middlefinger_left.cuda()
    limit_secondjoint_fourth_left = limit_secondjoint_fourth_left.cuda()
    limit_secondjoint_small_left = limit_secondjoint_small_left.cuda()

    limit_thirdjoint_bigfinger_left = limit_thirdjoint_bigfinger_left.cuda()
    limit_thirdjoint_index_left = limit_thirdjoint_index_left.cuda()
    limit_thirdjoint_middlefinger_left = limit_thirdjoint_middlefinger_left.cuda()
    limit_thirdjoint_fourth_left = limit_thirdjoint_fourth_left.cuda()
    limit_thirdjoint_small_left = limit_thirdjoint_small_left.cuda()


bigfinger_vertices = [697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
764, 765, 766, 767, 768]

indexfinger_vertices = [46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 133, 134, 155, 156, 164, 165, 166, 167, 174, 175, 189, 194, 195, 212, 213, 221, 222, 223, 224, 225, 226, 237, 238, 272, 273, 280, 281, 282, 283, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355]

middlefinger_vertices = [356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367,
372, 373, 374, 375, 376, 377, 381,
382, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
395, 396, 397, 398, 400, 401, 402, 403, 404, 405, 406, 407,
408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
460, 461, 462, 463, 464, 465, 466, 467]

fourthfinger_vertices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482, 483, 484, 485, 486, 487, 491, 492,
495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
508, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
573, 574, 575, 576, 577, 578]

smallfinger_vertices = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
598, 599, 600, 601, 602, 603,
609, 610, 613, 614, 615, 616, 617, 618, 619, 620,
621, 622, 623, 624, 625, 626, 628, 629, 630, 631, 632, 633,
634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
686, 687, 688, 689, 690, 691, 692, 693, 694, 695]



indexfinger_secondjoint_vertices = [221, 224, 237, 238, 272, 273, 281, 282, 283, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355]
indexfinger_thirdjoint_vertices = [304, 305, 306, 307, 308, 309, 311, 312, 313, 314, 315, 317, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 332, 333, 334, 335, 336, 337, 338, 339, 343, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355]

middlefinger_secondjoint_vertices = [390, 393,
396, 397, 398, 400, 401, 403, 404, 405, 406, 407,
408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
460, 461, 462, 463, 464, 465, 466, 467]
middlefinger_thirdjoint_vertices = [ 416, 417, 418, 419,
421, 423, 424, 425, 426, 428, 429, 432, 433,
434, 435, 436, 437, 438, 439, 442, 443, 444, 445, 446,
447, 448, 449, 450, 451, 455, 458, 459,
460, 461, 462, 463, 464, 465, 466, 467]

fourthfinger_secondjoint_vertices = [500, 503, 506, 507,
508, 511, 512, 514, 515, 516, 517, 518, 519, 520,
521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
573, 574, 575, 576, 577, 578]
fourthfinger_thirdjoint_vertices = [527, 528, 529, 530, 532, 533,
534, 535, 536, 537, 538, 539, 540, 543, 544, 545, 546,
547, 548, 549, 550, 553, 554, 555, 556, 557, 558, 559,
560, 561, 562, 566, 569, 570, 571, 572,
573, 574, 575, 576, 577, 578]

smallfinger_secondjoint_vertices = [618,
621, 624, 625, 626, 628, 629, 631, 632, 633,
634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
smallfinger_thirdjoint_vertices = [644, 645, 646,
651, 652, 653, 654, 656, 657,
660, 661, 662, 663, 664, 665, 666, 667, 670, 671, 672,
673, 674, 675, 676, 677, 678, 679,
686, 687, 688, 689, 690, 691, 692, 693, 694, 695]

bigfinger_secondjoint_vertices = [697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
764, 765, 766, 767, 768]
bigfinger_thirdjoint_vertices = [745, 744, 766, 729, 735, 751, 765, 730, 752, 764, 738, 728, 768,
       727, 767, 743, 747, 720, 748, 717, 750, 734, 761, 737, 724, 762,
       763, 726, 740, 719, 746, 718, 725, 722, 723, 733, 749, 716, 731,
       721, 736, 759, 739, 760, 756]

# Initialize MANO layer
MANO = ManoLayer(
    mano_root='/home/enric/libraries/manopth/mano/models/', side='right', use_pca=True, ncomps=45, flat_hand_mean=True)
#if torch.cuda.device_count() > 1:
    #print("Let's use", torch.cuda.device_count(), "GPUs!")
    #MANO = torch.nn.DataParallel(MANO)
MANO = MANO.cuda()
