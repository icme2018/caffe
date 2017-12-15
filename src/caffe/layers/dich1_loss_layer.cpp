#include <algorithm>
#include <vector>

#include "caffe/layers/dich1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

    template <typename Dtype>
    void DICH1LossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top)
    {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        CHECK_EQ(bottom[0]->height(), 1);
        CHECK_EQ(bottom[0]->width(), 1);
        CHECK_EQ(bottom[1]->height(), 1);
        CHECK_EQ(bottom[1]->width(), 1);
        diff_.Reshape(1, bottom[0]->channels(), 1, 1);
        // vector of ones used to sum along channels
        summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
        for (int i = 0; i < bottom[0]->channels(); ++i)
            summer_vec_.mutable_cpu_data()[i] = Dtype(1);

        tmpBottom_.Reshape(bottom[0]->shape());

    }

    template <typename Dtype>
    void DICH1LossLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*> &top)
    {
        // initialize parameters
        Dtype *bout = bottom[0]->mutable_cpu_diff(); 
        const int num = bottom[0]->num(); 
        const Dtype alpha = top[0]->cpu_diff()[0] / static_cast<Dtype>(num * (num - 1)); 
        const Dtype beta = top[0]->cpu_diff()[0] / static_cast<Dtype>(num);  
        const int channels = bottom[0]->channels();  
        Dtype margin = this->layer_param_.dich1_loss_param().bi_margin();  
        Dtype tradeoff = this->layer_param_.dich1_loss_param().tradeoff(); 
        Dtype scale = this->layer_param_.dich1_loss_param().scale();
        const int label_num = bottom[1]->count() / num;  
        bool sim;
        Dtype loss(0.0);
        Dtype reg(0.0);
        Dtype data(0.0);
        Dtype dist_sq(0.0);
        caffe_set(channels * num, Dtype(0), bout); 

        Dtype loss1(0.0);  // for our loss
        Dtype block_dist_sq(0.0);
        int cnt_dis = 0;  // count the dissimilar pair
        Dtype *bout1 = tmpBottom_.mutable_cpu_diff();
        caffe_set(channels * num, Dtype(0), bout1);

        // calculate loss and gradient
        for (int i = 0; i < num; ++i)
        {
            for (int j = i + 1; j < num; ++j)
            {
                caffe_sub(
                    channels,
                    bottom[0]->cpu_data() + (i * channels), // a
                    bottom[0]->cpu_data() + (j * channels), // b
                    diff_.mutable_cpu_data());  // a_i-b_i
                dist_sq = caffe_cpu_dot(channels, diff_.cpu_data(), diff_.cpu_data());  //D_w^2
                if (label_num > 1)
                {
                    sim = caffe_cpu_dot(label_num, bottom[1]->cpu_data() + (i * label_num), bottom[1]->cpu_data() + (j * label_num)) > 0;
                }
                else
                {
                    sim = ((static_cast<int>(bottom[1]->cpu_data()[i])) == (static_cast<int>(bottom[1]->cpu_data()[j])));
                }
                if (sim)    // similar pairs
                {
                    loss += dist_sq;
                    // gradient with respect to the first sample
                    caffe_cpu_axpby(
                        channels,
                        alpha,
                        diff_.cpu_data(),
                        Dtype(1.0),
                        bout + (i * channels));
                    // gradient with respect to the second sample
                    caffe_cpu_axpby(
                        channels,
                        -alpha,
                        diff_.cpu_data(),
                        Dtype(1.0),
                        bout + (j * channels));
                }
                else    // dissimilar pairs
                {
                    loss += std::max(margin - dist_sq, Dtype(0.0));
                    if ((margin - dist_sq) > Dtype(0.0))
                    {
                        // gradient with respect to the first sample
                        caffe_cpu_axpby(
                            channels,
                            -alpha,
                            diff_.cpu_data(),
                            Dtype(1.0),
                            bout + (i * channels));
                        // gradient with respect to the second sample
                        caffe_cpu_axpby(
                            channels,
                            alpha,
                            diff_.cpu_data(),
                            Dtype(1.0),
                            bout + (j * channels));
                    }

                    cnt_dis++;
                    for(int offset = 0; offset < channels; offset += 2){
                        block_dist_sq = caffe_cpu_dot(2, diff_.cpu_data()+offset, diff_.cpu_data()+offset);
                        loss1 += std::max(Dtype(4.0) - block_dist_sq, Dtype(0.0));
                        if(block_dist_sq < Dtype(4.0)){
                            caffe_cpu_axpby(2, Dtype(-1.0), diff_.cpu_data()+offset, Dtype(1.0), bout1+(i*channels)+offset);
                            caffe_cpu_axpby(2, Dtype( 1.0), diff_.cpu_data()+offset, Dtype(1.0), bout1+(j*channels)+offset);
                        }
                    }

                }
            }
            for (int k = 0; k < channels; k++)
            {
                data = *(bottom[0]->cpu_data() + (i * channels) + k);
                // gradient corresponding to the regularizer
                *(bout + (i * channels) + k) += beta * tradeoff * (((data >= Dtype(1.0)) || (data <= Dtype(0.0) && data >= Dtype(-1.0))) ? Dtype(1.0) : Dtype(-1.0));
                data = std::abs(data) - 1;
                reg += std::abs(data);
            }
        }
        loss = loss / static_cast<Dtype>(bottom[0]->num() * (bottom[0]->num() - 1));
        loss += tradeoff * reg / static_cast<Dtype>(bottom[0]->num());

        caffe_scal(channels * num, Dtype(1) / cnt_dis, bout1);
        loss += scale * loss1 / static_cast<Dtype>(cnt_dis);
        for(int c = 0; c < channels*num; c++){
            bout[c] += scale * bout1[c];
        }

        top[0]->mutable_cpu_data()[0] = loss;

    }

    template <typename Dtype>
    void DICH1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> &top,
            const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom)
    {
        if (propagate_down[1])
        {
            LOG(FATAL) << this->type()
                       << " Layer cannot backpropagate to label inputs.";
        }
    }

#ifdef CPU_ONLY
    //STUB_GPU(DICH1LossLayer);
#endif

    INSTANTIATE_CLASS(DICH1LossLayer);
    REGISTER_LAYER_CLASS(DICH1Loss);

}  // namespace caffe
