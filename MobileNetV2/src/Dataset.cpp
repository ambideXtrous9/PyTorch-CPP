#include "../inc/Dataset.h" 


CustomDataset::CustomDataset(const std::string &folderPath)
{
    for (const auto &entry : fs::directory_iterator(folderPath))
    {
        if (entry.is_directory())
        {
            // If the entry is a directory, iterate through its contents
            for (const auto &fileEntry : fs::directory_iterator(entry.path()))
            {
                if (fileEntry.is_regular_file() && fileEntry.path().extension() == ".jpg" )
                {
                    filepaths_.push_back(fileEntry.path());
                    // Assuming folder name is the label (0 for cat, 1 for dog)
                    if (entry.path().filename() == "Cat")
                    {
                        labels_.push_back(0);
                    }
                    else
                    {
                        labels_.push_back(1);
                    }
                }
            }
        }
    }
}

torch::data::Example<> CustomDataset::get(size_t index)
{
    cv::Mat img = cv::imread(filepaths_[index]);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(28, 28));
    torch::Tensor imageTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    imageTensor = imageTensor.permute({2, 0, 1}).to(torch::kFloat32); //.div_(255);
    int64_t label = labels_[index];
    return {imageTensor.clone(), torch::tensor(label)};
}

torch::optional<size_t> CustomDataset::size() const
{
    return filepaths_.size();
}