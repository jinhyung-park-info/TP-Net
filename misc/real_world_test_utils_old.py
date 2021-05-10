def compare_real_world_prediction_results(model_1, model_2, data_type1, data_type2, save_path, offset, fps):

    for case in REAL_WORLD_TEST_CASES:

        print(f'============== Testing Case #{case} ==============')

        output_video = cv2.VideoWriter(os.path.join(save_path, f'Test Case_{case}.MP4'), CODEC, fps, (VIDEO_HEIGHT * 3, VIDEO_HEIGHT))
        frames_savepath = create_directory(os.path.join(save_path, f'Test Case {case}'))
        ground_truth_base_path = f'D:\\WhereToGo\\real_world\\raw_video\\sample_{case}\\subtracted_frames-KNN\\cropped'

        input_info_1 = get_real_world_input_pointset(case, offset, data_type1)
        input_info_2 = get_real_world_input_pointset(case, offset, data_type2)
        first_frame_number, num_frames = find_case_info(ground_truth_base_path)

        frame = concat_three_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, 0, first_frame_number)
        for _ in range(20):
            output_video.write(frame)

        for timestep in range(0, NUM_INPUT_FRAMES * offset, offset):
            frame = concat_three_real_world_ground_truth_frames(ground_truth_base_path, frames_savepath, timestep, first_frame_number)
            output_video.write(frame)

        for timestep in tqdm(range(offset * NUM_INPUT_FRAMES, NUM_SEQUENCE_PER_ANIMATION, offset)):
            predicted_pointset_1 = model_1.predict(input_info_1)[0]     # Use only the first predicted frame
            predicted_pointset_2 = model_2.predict(input_info_2)[0]

            coordinates_1 = denormalize_pointset(predicted_pointset_1[0])
            coordinates_2 = denormalize_pointset(predicted_pointset_2[0])
            predicted_frame_1 = draw_box2d_image(coordinates_1)
            predicted_frame_2 = draw_box2d_image(coordinates_2)

            # concatenate with ground truth image for comparison
            merged_frame = concat_multiple_preds_and_real_world_gt_frame(predicted_frame_1, predicted_frame_2, ground_truth_base_path, timestep, frames_savepath, first_frame_number, num_frames)
            output_video.write(merged_frame)
            input_info_1 = update_input_pointset(input_info_1, predicted_pointset_1)
            input_info_2 = update_input_pointset(input_info_2, predicted_pointset_2)

        output_video.release()
        cv2.destroyAllWindows()
