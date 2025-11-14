#pragma once

#ifdef BUILDING_FOR_WINDOWS
    #define EXPORT __declspec( dllexport )
#else
    #define EXPORT
#endif
