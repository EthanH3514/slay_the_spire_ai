#pragma once
// Opt-in inference only. No simulator state, RNG, search backup or global cache.
#include "LeafActorFast.h"

namespace spire_policy_value {
using namespace spire_actor;
inline constexpr auto version = "public-combat-policy-value-v1";
inline constexpr unsigned quantileCount = 21;

template<unsigned I, unsigned O> struct Linear {
    std::array<float,I*O> weight; std::array<float,O> bias;
    void load(const J& tensors, const std::string& name) {
        auto read = [&](const std::string& key, const J& shape, auto& target) {
            const auto& t=tensors.at(key); keys(t,{"shape","values"},{"shape","values"});
            if(t.at("shape")!=shape || !t.at("values").is_array() || t.at("values").size()!=target.size())
                throw std::invalid_argument("value_tensor_shape");
            unsigned i=0;
            for(const auto& x:t.at("values")) {
                if(!x.is_number() || !std::isfinite(x.get<double>()) || std::abs(x.get<double>())>std::numeric_limits<float>::max())
                    throw std::invalid_argument("value_tensor_finite");
                target[i++]=x.get<float>();
            }
        };
        read(name+".weight",J::array({O,I}),weight); read(name+".bias",J::array({O}),bias);
    }
    std::array<float,O> run(const std::array<float,I>& x,bool activate=false) const {
        std::array<float,O> y;
        for(unsigned i=0;i<O;++i) {
            float sum=bias[i]; for(unsigned j=0;j<I;++j)sum+=weight[i*I+j]*x[j];
            y[i]=activate?std::tanh(sum):sum;
            if(!std::isfinite(y[i]))throw std::runtime_error("value_nonfinite_output");
        }
        return y;
    }
};

struct Prediction {
    std::vector<float> logits;
    std::array<float,2> outcomeLogits;
    std::array<std::array<float,quantileCount>,2> quantiles;
    J json() const {
        const double top=std::max(outcomeLogits[0],outcomeLogits[1]);
        const double a=std::exp(outcomeLogits[0]-top),b=std::exp(outcomeLogits[1]-top);
        return J{{"logits",logits},{"outcome_logits",outcomeLogits},
                 {"outcome_probabilities",J::array({a/(a+b),b/(a+b)})},{"resource_quantiles",quantiles}};
    }
};

class Model {
    spire_actor_fast::Actor encoder_;
    std::vector<float> embedding_;
    Linear<96,32> row_; Linear<396,32> context_; Linear<164,32> head_; Linear<32,1> output_;
    Linear<32,64> hidden_; Linear<64,2> outcome_; Linear<64,2*quantileCount> resource_;
public:
    // The caller supplies the independently pinned actor and its canonical digest.
    // Full actor equality prevents substituting a different behavior/model while
    // merely relabelling a digest. Python verifies the canonical SHA at the boundary.
    Model(const J& value,const J& expectedActor,const std::string& expectedDigest)
        :encoder_(expectedActor,expectedActor.at("schema")) {
        const std::set<std::string> fields={"version","actor","value_tensors","dtype","target","outcome_order",
            "quantile_levels","resource_scale","value_gradient_to_encoder","continuation"};
        keys(value,fields,fields);
        J levels=J::array();for(unsigned i=0;i<quantileCount;++i)levels.push_back((i+.5)/quantileCount);
        const J continuation={{"kind","actor_softmax"},{"temperature",1.},{"policy_digest",expectedDigest}};
        if(value.at("version")!=version || value.at("actor")!=expectedActor || expectedDigest.size()!=64 ||
           value.at("dtype")!="float32" || value.at("target")!="terminal-outcome-resource-quantiles-v1" ||
           value.at("outcome_order")!=J::array({"loss","win"}) || value.at("quantile_levels")!=levels ||
           value.at("resource_scale")!=100. || value.at("value_gradient_to_encoder")!=false ||
           value.at("continuation")!=continuation)throw std::invalid_argument("value_artifact_semantics");
        const auto& v=value.at("value_tensors");
        const std::set<std::string> names={"value_hidden.weight","value_hidden.bias","outcome.weight","outcome.bias","resource.weight","resource.bias"};
        keys(v,names,names);
        hidden_.load(v,"value_hidden");outcome_.load(v,"outcome");resource_.load(v,"resource");
        const auto& p=expectedActor.at("tensors");
        embedding_=p.at("embedding.weight").at("values").get<std::vector<float>>();
        row_.load(p,"row");context_.load(p,"context");head_.load(p,"head");output_.load(p,"output");
    }
    Encoded encode(const J& view,const J& legal) const {return encoder_.encode(view,legal);}
    Prediction forward(const Encoded& encoded) const {
        std::map<Row,std::array<float,32>> cache;
        auto pool=[&](const Group& group) {
            std::array<float,33> out{};
            for(const auto& r:group) {
                auto it=cache.find(r);
                if(it==cache.end()) {
                    std::array<float,96> x;unsigned slot=0;
                    for(int token:r.categories) {
                        if(token<0 || std::size_t(token)*8+8>embedding_.size())throw std::invalid_argument("value_token_range");
                        for(unsigned j=0;j<8;++j)x[slot++]=embedding_[token*8+j];
                    }
                    for(float v:r.values)x[slot++]=v;for(float v:r.known)x[slot++]=v;
                    it=cache.emplace(r,row_.run(x,true)).first;
                }
                for(unsigned i=0;i<32;++i)out[i]+=it->second[i];
            }
            if(!group.empty()) {
                for(unsigned i=0;i<32;++i)out[i]/=float(group.size());
                out[32]=float(std::log1p(double(group.size())));
            }
            return out;
        };
        if(encoded.state.size()!=12)throw std::invalid_argument("value_state_shape");
        std::array<float,396> pooled;unsigned offset=0;
        for(const auto& group:encoded.state) {
            const auto p=pool(group);std::copy(p.begin(),p.end(),pooled.begin()+offset);offset+=33;
        }
        const auto context=context_.run(pooled,true);Prediction out;
        std::map<std::array<float,164>,float> candidates;
        for(const auto& candidate:encoded.candidates) {
            if(candidate.size()!=4)throw std::invalid_argument("value_candidate_shape");
            std::array<float,164> x;std::copy(context.begin(),context.end(),x.begin());offset=32;
            for(const auto& group:candidate) {
                const auto p=pool(group);std::copy(p.begin(),p.end(),x.begin()+offset);offset+=33;
            }
            auto it=candidates.find(x);
            if(it==candidates.end())it=candidates.emplace(x,output_.run(head_.run(x,true))[0]).first;
            out.logits.push_back(it->second);
        }
        const auto hidden=hidden_.run(context,true);out.outcomeLogits=outcome_.run(hidden);
        const auto raw=resource_.run(hidden);
        for(unsigned o=0;o<2;++o) {
            float sum=0;
            for(unsigned i=0;i<quantileCount;++i) {
                const float x=raw[o*quantileCount+i];
                sum+=x>20.f?x:std::log1p(std::exp(x));
                if(!std::isfinite(sum))throw std::runtime_error("value_nonfinite_quantile");
                out.quantiles[o][i]=sum;
            }
        }
        return out;
    }
};
} // namespace spire_policy_value
