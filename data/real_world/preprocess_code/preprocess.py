from data.real_world.preprocess_code.preprocess_utils import *

if __name__ == '__main__':

    """
    video_numbers = [i for i in range(1, 73)]
    video_numbers.remove(34)
    video_numbers.remove(47)
    
    video_to_image(video_numbers=video_numbers,
                   start_times=[94, 102, 86, 78, 112, 109, 98, 101, 97, 80,
                                96, 88, 89, 82, 105, 92, 96, 95, 96, 120,
                                92, 98, 97, 99, 100, 95, 101, 98, 102, 106,
                                86, 88, 96, 102, 90, 88, 98, 97, 89,
                                96, 97, 93, 102, 94, 86, 85, 83, 93,
                                98, 104, 91, 80, 76, 91, 82, 80, 83, 87,
                                86, 78, 78, 84, 87, 86, 95, 78, 94, 77,
                                93, 86])
    crop_images([1], [1734], [934])
    crop_images([2], [1626], [926])
    crop_images([3, 6], [1624] * 2, [922] * 2)
    crop_images([4], [1624], [948])
    crop_images([5], [1624], [943])
    crop_images([7], [1624], [948])
    crop_images([8], [1632], [934])
    crop_images([9], [1632], [955])
    crop_images([i for i in range(10, 13)], [1630] * 3, [943] * 3)

    crop_images([13], [1626], [952])
    crop_images([14], [1640], [953])
    crop_images([i for i in range(15, 18)], [1624] * 3, [941] * 3)

    crop_images([18], [1640], [943])
    crop_images([19], [1624], [941])
    numbers = [i for i in range(20, 29)]
    numbers.remove(24)
    numbers.remove(25)
    numbers.remove(28)
    crop_images(numbers, [1562] * 6, [943] * 6)

    crop_images([24], [1584], [953])
    crop_images([25], [1584], [943])
    crop_images([28], [1562], [931])

    crop_images([29], [1580], [939])
    crop_images([30], [1496], [930])
    crop_images([31], [1496], [924])
    crop_images([32], [1478], [924])
    crop_images([33], [1492], [938])
    crop_images([i for i in range(35, 42)], [1472] * 7, [929] * 7)
    crop_images([i for i in range(42, 47)], [1470] * 5, [926] * 5)

    #crop_images([48], [1494], [941])
    #crop_images([49], [1490], [938])
    """

    #image_background_subtraction(video_numbers=[51], algorithm='KNN')
    #image_background_subtraction(video_numbers=[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    #                                            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72], algorithm='KNN')
    #video_numbers = [i for i in range(45, 47)]
    #video_numbers.remove(34)

    #video_numbers = [i for i in range(1, 47)]
    #video_numbers.remove(34)
    #video_numbers = [i for i in range(48, 73)]
    #video_numbers.remove(53)
    #video_numbers.remove(55)
    #video_numbers.remove(58)
    #video_numbers.remove(59)
    #video_numbers.remove(65)
    #crop_images(video_numbers=video_numbers)
    #

    #sequences = [(78, 253), (92, 293), (83, 244), (52, 235), (66, 244), (92, 297), (96, 262), (76, 336), (101, 279), (109, 334), (62, 242),
    #             (96, 320), (120, 319), (94, 346), (109, 317),
    #             (83, 295), (70, 282), (71, 328), (79, 336), (91, 334)]
    #remove_unnecessary_frames(video_numbers=video_numbers, sequences=sequences)

    #get_point_cloud_data(video_numbers=video_numbers)

    #generate_prediction_model_data(num_input=3, num_output=8, offset=2)
    image_to_video(frames_paths=['C:\\Users\\User\\Desktop\\shared\\WhereToGo\\result\\global_pointnet\\version_13\\fine_tuning_result_rendered\\offset_2_version_3\\Test Case 63'])
