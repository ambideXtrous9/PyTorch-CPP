#include "../inc/prepDataset.h"

bool isCorrupt(const fs::path &filePath)
{
    try
    {
        cv::Mat image = cv::imread(filePath.string());
        if (image.empty() || !image.data) // Check if the image is empty
        {
            return true; // Image is considered corrupt
        }
    }
    catch (cv::Exception &e)
    {
        return true; // Exception occurred while reading the image, considered corrupt
    }
    return false; // Image is not corrupt
}

void createTrainTestFolders(const std::string &parentFolder, const std::string &categoryFolder, const std::string &trainFolder, const std::string &testFolder, double trainRatio)
{
    fs::path catFolder = fs::path(parentFolder) / categoryFolder;
    fs::path trainCatFolder = fs::path(parentFolder) / trainFolder / categoryFolder;
    fs::path testCatFolder = fs::path(parentFolder) / testFolder / categoryFolder;

    // Create train and test folders if they don't exist
    fs::create_directories(trainCatFolder);
    fs::create_directories(testCatFolder);

    // Collect all image files
    std::vector<fs::path> imageFiles;
    for (const auto &entry : fs::directory_iterator(catFolder))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg" && !isCorrupt(entry.path()))
        {

            imageFiles.push_back(entry.path());
        }
    }

    // Shuffle the image files
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(imageFiles.begin(), imageFiles.end(), g);

    // Determine number of images for training
    size_t numTrainImages = static_cast<size_t>(trainRatio * imageFiles.size());

    // Copy images to train folder
    for (size_t i = 0; i < numTrainImages; ++i)
    {

        fs::copy_file(imageFiles[i], trainCatFolder / imageFiles[i].filename(), fs::copy_options::overwrite_existing);
    }

    // Copy remaining images to test folder
    for (size_t i = numTrainImages; i < imageFiles.size(); ++i)
    {

        fs::copy_file(imageFiles[i], testCatFolder / imageFiles[i].filename(), fs::copy_options::overwrite_existing);
    }
}
