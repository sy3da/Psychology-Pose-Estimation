#include <iostream>
#include <regex>
#include <random>
#include <glog/logging.h>
#include <thread>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace std::chrono;

#include "tof_framework_api.hpp"
#include "sigslot.h"
#include "tof_utils.hpp"
#include "file_reader.hpp"

struct Listen: public sigslot::has_slots {

};

#define MEASURE_AND_DISPLAY_FRAMERATE

bool need_to_save_depth=false;
bool need_to_save_confidence=false;
bool need_to_save_amplitude=false;
bool need_to_save_ambient = false;
Size selected_size;

// Due to the linker when the static library is used by the application, the pipeline is never registered (constructor never called automatically)
// Unfortunately it's necessary to do it on the application side in order to have a reference and avoid the linker removal
#include "thanos_pipeline_handler.hpp"
REGISTER_PIPELINE_HANDLER(ThanosPipelineHandler)


TOFManager tof_manager;

// UI Helpers 
void FillDescriptionVector(std::vector<Size> & sizes, std::vector<std::string> & descriptions)
{
    for(auto size : sizes) {
        descriptions.push_back(size.getLabel());
    }
}

void FillDescriptionVector(std::vector<HardwareAccelerator> & accelerators, std::vector<std::string> & descriptions)
{
    for(auto accelerator : accelerators) {
        std::string desc=accelerator.getName();
        descriptions.push_back(desc);
    }
}

void FillDescriptionVector(std::vector<StreamConfig> & configs, std::vector<std::string> & descriptions)
{
    for(auto config : configs) {
        std::string desc=config.getPixelFormat().getLabel()+"-"+config.getSize().getLabel();
        descriptions.push_back(desc);
    }
}

void FillDescriptionVector(std::vector<StreamProfile> & profiles, std::vector<std::string> & descriptions)
{
    for(auto profile : profiles) {
        descriptions.push_back(profile.getDescription());
    }
}

void FillDescriptionVector(std::vector<std::string> & profiles, std::vector<std::string> & descriptions)
{
    descriptions=profiles;
}

template<typename T>
int displayValuesAndPickOne( std::vector<T> & allItems, char * itemName="item")
{
    int selected=-1;

    std::vector<std::string> descriptions;
    FillDescriptionVector(allItems, descriptions);

    std::cout << "\nSelect " << itemName << ": \n";
    for(size_t i = 0; i < descriptions.size(); ++i)
    {
         std::cout << i << " - " << descriptions[i] << '\n';
    }   
    std::cout << '\n' << "Type number of desired value & press Enter.\n";
    std::cin >> selected;

    if(selected < 0)
        selected=0;
    if(selected > allItems.size()-1)
        selected=allItems.size()-1;

    return selected;
}

/* TOF management Functions */
const std::shared_ptr<TOF> initTOF()
{
    if(tof_manager.initialize() != TOF_MANAGER_NO_ERROR) {
        std::cerr << "Failed to initialize";
        return nullptr;
    }
    
    // Get and return the TOF corresponding to the VD55H1 sensor
    return tof_manager.getTOFByName("VD55H1");
}

void requestReadyFunction(Request* request) 
{
    std::map<std::string, std::vector<Buffer>> stream_buffers_map = request->getResultBuffers();                                                    
    const uint16_t * depth = nullptr;
    const float * amplitude = nullptr;
    const float * confidence = nullptr;
    const float * ambient = nullptr;
    int size = 0;
    Buffer::Metadata  metadata;

 #ifdef MEASURE_AND_DISPLAY_FRAMERATE 
    using namespace std::chrono;
 
    static high_resolution_clock::time_point last_time= high_resolution_clock::now();
    static int frame_count=0;
    static int milliseconds_count=1;
    static double frame_rate=-1;
    frame_count++;
    duration<double, std::milli> time_spent = high_resolution_clock::now() - last_time;    
    milliseconds_count+=time_spent.count();
    last_time=high_resolution_clock::now();

    if(milliseconds_count > 1000) // Frame rate is updated and displayed only once per seconds
    {
        frame_rate=1000.0*(double)frame_count/(double)milliseconds_count;
        milliseconds_count=0;
        frame_count=0;
        for(int i=0; i <25; i++)
            std::cout  << "\b";    // delete previously displayed text
        std::cout << "Frame rate: " <<  std::fixed << std::setprecision(2) << frame_rate << " fps";
    }
#endif

    for( auto const& [stream_id, buffers] : stream_buffers_map ) {                                                        
        if(!buffers.empty()) {
            Buffer buffer= buffers.at(0);
            if(buffer.m_metadata)
                metadata= buffer.m_metadata.value();
            else
                std::cerr << "Invalid Metadata";

            if(!stream_id.compare("Depth") && need_to_save_depth) {
                depth = static_cast<const uint16_t *>(buffer.m_buffer);
                FileReader::savePgmFile("depth.pgm", (char*)depth, buffer.m_size, 
                            selected_size.getWidth(), selected_size.getHeight(), 
                            8191, 
                            "# Depth saved with tuto app\n");
                need_to_save_depth=false;
            } else if(!stream_id.compare("Amplitude") && need_to_save_amplitude) {
                amplitude = static_cast<const float *>(buffer.m_buffer);
                FileReader::savePfmFile("amplitude.pfm", (char*)amplitude, buffer.m_size, 
                            selected_size.getWidth(), selected_size.getHeight(), 
                            pow(2,12+log2( metadata.xScale))-1); 
                need_to_save_amplitude=false;
            }
            else if(!stream_id.compare("Confidence") && need_to_save_confidence) {
                confidence = static_cast<const float *>(buffer.m_buffer);
                FileReader::savePfmFile("confidence.pfm", (char*)confidence, buffer.m_size, 
                            selected_size.getWidth(), selected_size.getHeight(), 
                            pow(2,12+log2( metadata.xScale))-1); 
                need_to_save_confidence=false;
            }
            else if(!stream_id.compare("Ambient") && need_to_save_ambient) {
                ambient = static_cast<const float *>(buffer.m_buffer);
                FileReader::savePfmFile("ambient.pfm", (char*)ambient, buffer.m_size, 
                            selected_size.getWidth(), selected_size.getHeight(), 
                            pow(2,12+log2( metadata.xScale))-1); 
                need_to_save_ambient=false;
            }
        }
        
        // Need to free all buffers from framework after use 
        for(auto const & _buffer: buffers) {
            try { 
                if(_buffer.m_flags & Buffer::FLAG_ALIGNED_MALLOC)
                    TofUtils::aligned_free(_buffer.m_buffer);
                else                                                              
                    free(_buffer.m_buffer);                                                                            
            } 
            catch(std::exception exception) {
                    std::cerr<< "Exception while freeing buffer: " << exception.what(); 
            }
        }
    }
} 

void errorFunction(int error_code) 
{
     std::cerr<< "Error " << error_code << " occured during processing\n";
}

void setOutputResolution(std::vector<StreamConfig> & stream_configs)
{
   // Change output resolution for the configuration. We need do it for all relevant streams
    int index_resolution = -1;
    for(auto it = std::begin(stream_configs); it != std::end(stream_configs); ++it) {
        std::string stream_name = it->getStream()->getName();
        std::vector<Size> size_capabilities = it->getSizes(it->getPixelFormat());
        // We need to change the size for all streams except the Sensor one
        if(stream_name.compare("Sensor")!=0) {  
            if(index_resolution == -1)  {  // Select size only once since it will be the same for all
                index_resolution=displayValuesAndPickOne(size_capabilities, "an output resolution");
            }
            //Set Resolution Output resolution
            //stream_config.setSize(size_capabilities.at(size_capabilities.size()-1));           // Take last available compatinility by default            
            if(size_capabilities.size()>index_resolution) {
                selected_size=size_capabilities.at(index_resolution);
                it->setSize(selected_size);
            }
        }
    }  
}

int main(int argc, char* argv[]) 
{
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 1;    // Remove INFO traces from framework if any

    // Google logging is used by framework to output traces
    google::InitGoogleLogging(argv[0]);

    std::cout << "Initialising TOF...\n";
    const std::shared_ptr<TOF> tof = initTOF();
    if(tof == nullptr)
        exit(-1);

    // Get all supported profiles
    std::vector<StreamProfile> profiles = tof->getProfiles();
    // Ask for a profile selection
    StreamProfile selectedProfile=profiles.at(displayValuesAndPickOne(profiles, "a profile"));
    
    // Build the default configuration of all streams based on selected profile
    std::vector<StreamConfig> stream_configs = tof->buildConfiguration(selectedProfile); // StreamProfile::FULL_DEPTH_3DRAW_3F_LR_BIN);  
    // Select and change output resolution
    setOutputResolution(stream_configs);

    // Let's configure the streams with the modified configuration
    tof->setStreamConfiguration(stream_configs);

    // Set specific controls for sensor/pipeline
    
    // Enable DarkCal config
    tof->setControls({new Control<PapaBearDarkCalConfig>(DARKCAL_CONTROL_ID, PapaBearDarkCalConfig(true, 0x7, 0x40))});

    // First exposure
    #define DEFAULT_EXPOSURE 0.35
    std::cout << "\nSetting exposure to " << std::setprecision(2) << DEFAULT_EXPOSURE<< "\n";
    tof->setControls({new Control<double>(EXPOSURE_CONTROL_ID, DEFAULT_EXPOSURE)});

    // Verify that set control value is correct
    Control<double> expoControl(EXPOSURE_CONTROL_ID,0.0);
    std::vector<IControlId*> controlsVector;
    controlsVector.push_back(&expoControl);
    if(tof->getControls(&controlsVector)==0)
        std::cout << "Exposure was set to " << std::setprecision(2) << expoControl.getValue() << " milliseconds \n";
    else
        std::cerr << "Failed to get exposure control\n";

    Request * request = tof->createRequest();

    // Per stream buffer allocation
    std::map<Stream*, std::vector<std::unique_ptr<Buffer>>> buffer_map;
    for(StreamConfig stream_config : stream_configs) {
        std::vector<std::unique_ptr<Buffer>> stream_buffers;
        if(!tof->allocateBuffers(stream_config.m_stream, &stream_buffers)) {
            request->addBuffers(stream_config.m_stream, stream_buffers);
            buffer_map[stream_config.m_stream] = std::move(stream_buffers);
        }
        else {
             std::cerr << "Failed to allocate buffers for " << stream_config.m_stream;
        }
    }
  
    std::vector<HardwareAccelerator> hardware_accelerators = tof->getHardwareAccelerators();
    int hardware_accelerators_index=0;    
    if(hardware_accelerators.size() > 1)
        hardware_accelerators_index=displayValuesAndPickOne(hardware_accelerators, "an Accelerator");
    tof->selectHardwareAccelerator(hardware_accelerators.at(hardware_accelerators_index));

    // Set the request lambda expression that will create and queue a new request each time a queue request signal is emit (i.e we can ask for a new frame)
    Listen listen;
    // tof->getQueueRequestSignal().connect(&listen, [tof](Request* request) {
    //                                                         // Post a new Request
    //                                                         Request * new_request = tof->createRequest();
    //                                                         // Reuse buffers of finished request for next one
    //                                                         for( auto & [_stream, _buffers] : request->getStreamBuffers() ) 
    //                                                             new_request->addBuffers(_stream, _buffers); 
    //                                                         tof->queueRequest(new_request);
    //                                             } );

    tof->getQueueRequestSignal().connect(&listen, [tof](Request* request) {
                                                            // Post a new Request
                                                            Request * new_request =tof->createRequest();
                                                           request->copyRequest(new_request);
                                                           tof->queueRequest(new_request);
                                                } );

    // Setup the connection to get buffers when they are ready. Here we use an external function instead of a lambda 
    tof->getRequestReadySignal().connect(&listen, requestReadyFunction);

    // Setup the connection to get buffers when they are ready. Here we use an external function instead of a lambda 
    tof->getOnErrorSignal().connect(&listen, errorFunction);

    // Now we can start the capture
    tof->startCapture();
    
    // And queue a first request to start the acquisition & processing
    tof->queueRequest(request);

    bool do_stop=false;
    do {
        std::cout << '\n' << "Press q+Enter to exit or d+Enter to dump data into files...\n";
        char c;
        std::cin >> c;
        switch (c)        
        {   
            case 'd':
                // Change this to select which file to save. By default, only depth and confidence are enabled
                need_to_save_depth=true;
                need_to_save_confidence=true;
                need_to_save_amplitude=false;
                need_to_save_ambient = false;
                std::cout << "Dumping data into files...\n";
                break;
            case 'q':
                std::cout << "Exiting...\n";
                do_stop=true;
        }
    } while (!do_stop);

    tof->stopCapture();

    tof_manager.release();
       
    return 0;
}


