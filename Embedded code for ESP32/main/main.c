/*
 * SPDX-FileCopyrightText: 2022-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "soc/soc_caps.h"
#include "esp_log.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"
#include "sdkconfig.h"
#include "time.h"
#include "freertos/FreeRTOSConfig.h"

#include "feature.h"
#include "distant.h"

const static char *TAG = "EXAMPLE";
#define BLINK_GPIO GPIO_NUM_2
#define EXAMPLE_ADC_ATTEN           ADC_ATTEN_DB_11
// sáng tiếp 2s ke tu lan phat hien chuyen dong cuoi cung
#define DURATION 2000
// bien dau ra cua du doan, 0 (den tat) va 1 (den sang)
static int state_predict = 0;


// Khai báo Semaphore
SemaphoreHandle_t data_ready_sem;
SemaphoreHandle_t led;
SemaphoreHandle_t check_led;

// cấu hình kích thước dữ liệu thu được
#define DEFAULT_BUOC_NHAY 50
#define DEFAULT_SIZE_INPUT 200
#define DEFAULT_SIZE_FEATURE matrix_size

// Kiểm tra nếu các giá trị không được chỉ định, sử dụng giá trị mặc định
#ifndef BUOC_NHAY
#define BUOC_NHAY DEFAULT_BUOC_NHAY
#endif

#ifndef SIZE_INPUT
#define SIZE_INPUT DEFAULT_SIZE_INPUT
#endif

#ifndef SIZE_FEATURE
#define SIZE_FEATURE DEFAULT_SIZE_FEATURE
#endif

static int adc_raw[BUOC_NHAY] = {0};
static int predict[SIZE_INPUT] = {0};
static float features[SIZE_FEATURE] = {0};

float distant_min_W0;
float distant_min_W1;

static char *current_time;

char *get_current_time_string() {
    time_t now;
    struct tm timeinfo;
    char *time_string = malloc(20 * sizeof(char)); // 20 bytes cho chuỗi thời gian

    // Lấy thời gian hiện tại
    time(&now);

    // // Chuyển đổi thời gian thành struct tm (tại múi giờ UTC)
    // gmtime_r(&now, &timeinfo);

    // Chuyển đổi thời gian thành struct tm
    localtime_r(&now, &timeinfo);

    // Tạo chuỗi thời gian
    strftime(time_string, 20, "%Y-%m-%d %H:%M:%S", &timeinfo);

    return time_string;
}


// task1: đọc dữ liệu từ cảm biến
void adc_read(void *pvParameters)
{
    //-------------ADC1 Init---------------//
    adc_oneshot_unit_handle_t adc1_handle;
    adc_oneshot_unit_init_cfg_t init_config1 = {
        .unit_id = ADC_UNIT_1,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config1, &adc1_handle));

    //-------------ADC1 Config---------------//
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_12,
        .atten = EXAMPLE_ADC_ATTEN,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, ADC_CHANNEL_4, &config));

    while (1) {
        int k = 0;
        // cập nhật (size - buoc_nhay) giá trị đầu trong predict
        for(int m = 0; m < (SIZE_INPUT - BUOC_NHAY); m ++){
            predict[m] = predict[m + BUOC_NHAY];
        }
        // cập nhật buoc_nhay giá trị mới nhất vào cuối predict memmove(pointer, pointer, bytes)
        for (int n = (SIZE_INPUT - BUOC_NHAY); n<SIZE_INPUT; n++){
            predict[n]= adc_raw[n-(SIZE_INPUT - BUOC_NHAY)];
        }
        // Khi có dữ liệu mới, gửi Semaphore để thông báo Task 2
        xSemaphoreGive(data_ready_sem);
        vTaskDelay(pdMS_TO_TICKS(10));
        while(1){
            // Chờ semaphore từ task3
            if (xSemaphoreTake(check_led, portMAX_DELAY) == pdTRUE){
                for(int i = 0; i < BUOC_NHAY; i++){
                    ESP_ERROR_CHECK(adc_oneshot_read(adc1_handle, ADC_CHANNEL_4, &adc_raw[i]));
                    current_time = get_current_time_string();
                    ESP_LOGI(TAG, "%s ADC%d Channel[%d] Raw Data: %d\t%d", current_time, ADC_UNIT_1 + 1, ADC_CHANNEL_4, adc_raw[i], k);
                    k++;
                    if(i != (BUOC_NHAY - 1)){
                        vTaskDelay(pdMS_TO_TICKS(10)); //block 10ms
                    }                  
                }
                break;
            }
        }       
    }

    //Tear Down
    ESP_ERROR_CHECK(adc_oneshot_del_unit(adc1_handle));
}


//task2: điều khiển led
void process_data(void *pvParameters){
    // Thiết lập chân GPIO cho đèn LED là đầu ra
    gpio_set_direction(BLINK_GPIO, GPIO_MODE_OUTPUT);
    while(1){
        // Chờ Semaphore từ Task 1
        if (xSemaphoreTake(data_ready_sem, portMAX_DELAY) == pdTRUE) {
            features[0] = mean(predict, SIZE_INPUT);
            features[1] = minimum(predict, SIZE_INPUT);
            features[2] = maximum(predict, SIZE_INPUT);
            features[3] = standard_deviation(predict, SIZE_INPUT);
            features[4] = slope(predict, SIZE_INPUT);
            distant_min_W0 = distant_min(features, W0, omega);
            distant_min_W1 = distant_min(features, W1, omega);

            // neu distant_min_W0 < distant_min_W1 thi state_predict la 0, nguoc lai la 1;
            //  khi distant...0 va distant...1 bang nhau thi state_predict giu nguyen gia tri truoc do
            if(distant_min_W0 < distant_min_W1)
                state_predict = 0;
            if(distant_min_W0 > distant_min_W1){
                // Bật đèn LED
                gpio_set_level(BLINK_GPIO, 1);
                ESP_LOGI(TAG,"led turn on");
                state_predict = 1;
            }
                
            // Gửi semaphore cho task3 để kiểm tra trạng thái đèn
            
            xSemaphoreGive(led); 
            
        }
    }
}

// task3
void led_control(void *pvParameters){
    TickType_t last_led_on_time = 0;
    const TickType_t LED_ON_DURATION = DURATION / portTICK_PERIOD_MS; // 2 seconds in ticks


    while (1) {
        // Chờ Semaphore từ Task 2
        if (xSemaphoreTake(led, portMAX_DELAY) == pdTRUE){
            // Kiểm tra nếu đèn được bật
            if (state_predict == 1) {
                // Lưu thời điểm đèn được bật cuối cùng
                last_led_on_time = xTaskGetTickCount();
                ESP_LOGI(TAG,"led is on");

            }

            // Kiểm tra nếu đã qua thời gian duy trì đèn sáng
            if (state_predict == 0 && (xTaskGetTickCount() - last_led_on_time >= LED_ON_DURATION)) {
                // Tắt đèn
                gpio_set_level(BLINK_GPIO, 0);
                ESP_LOGI(TAG,"led turn off");
            }

            // check led xong --> gửi semaphore cho task 1
            xSemaphoreGive(check_led); 
        }
    }
}
void app_main(){

        // Cấu hình chân GPIO là đầu ra
    gpio_set_direction(BLINK_GPIO, GPIO_MODE_OUTPUT);
    // ban dau den tat
    gpio_set_level(BLINK_GPIO, 0);
    vTaskDelay(pdMS_TO_TICKS(1000));


    // Khởi tạo Semaphore
    data_ready_sem = xSemaphoreCreateBinary();
    led = xSemaphoreCreateBinary();
    check_led = xSemaphoreCreateBinary();
    // gửi cho task1 bắt chạy
    xSemaphoreGive(check_led); 
    xTaskCreate(&adc_read, "adc_read", 2048, NULL, 10, NULL);
    xTaskCreate(&led_control, "led_control", 2048, NULL, 10, NULL );
    xTaskCreate(&process_data, "process_data", 2048, NULL, 10, NULL);


}

