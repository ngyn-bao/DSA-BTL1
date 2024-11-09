
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include "include/list/listheader.h"
#include "include/list/XArrayListDemo.h"
#include "include/list/DLinkedListDemo.h"
#include "include/ann/DataLoaderDemo.h"

using namespace std;

int main(int argc, char **argv)
{
    cout << "Assignment-1" << endl;
    string type = argv[1];
    if (type == "-xlist")
    {
        switch (argv[2][0])
        {
        case '1':
            xlistDemo1();
            break;
        case '2':
            xlistDemo2();
            break;
        case '3':
            xlistDemo3();
            break;
        case '4':
            xlistDemo4();
            break;
        default:
            xlistDemo1();
            xlistDemo2();
            xlistDemo3();
            xlistDemo4();
        }
    }
    if (type == "-dlist")
    {
        /* code */
        switch (argv[2][0])
        {
        case '1':
            dlistDemo1();
            break;
        case '2':
            dlistDemo2();
            break;
        case '3':
            dlistDemo3();
            break;
        case '4':
            dlistDemo4();
            break;
        case '5':
            dlistDemo5();
            break;
        case '6':
            dlistDemo6();
            break;
        default:
            dlistDemo1();
            dlistDemo2();
            dlistDemo3();
            dlistDemo4();
            dlistDemo5();
            dlistDemo6();
        }
    }
    if (type == "-dataloader")
    {
        /* code */
        switch (argv[2][0])
        {
        case '1':
            dataloadertc1();
            break;
        case '2':
            dataloadertc2();
            break;
        case '3':
            dataloadertc3();
            break;
        case '4':
            dataloadertc4();
            break;
        case '5':
            dataloadertc5();
            break;
        case '6':
            dataloadertc6();
            break;
        case '7':
            dataloadertc7();
            break;
        case '8':
            case_data_wo_label_1();
            break;
        case '9':
            case_data_wi_label_1();
            break;
        case '0':
            case_batch_larger_nsamples();
            break;

        default:
            dataloadertc1();
            dataloadertc2();
            dataloadertc3();
            dataloadertc4();
            dataloadertc5();
            dataloadertc6();
            dataloadertc7();
            case_data_wo_label_1();
            case_data_wi_label_1();
            case_batch_larger_nsamples();
        }
    }
    return 0;
}

// g++ -std=c++17 -I "D:\Dai hoc\241\CTDL&GT\BTL1\include" -Iinclude -Isrc -o main main.cpp; ./main.exe