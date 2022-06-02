#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <chrono>
using namespace std;

int main(int argc, char** argv)
{
    int conn_sock;
    struct sockaddr_in server_addr;

    int num = atoi(argv[3]);
    int port = atoi(argv[2]);
    char* addr = argv[1];

    if (port < 0 || port > 60000)
        exit(3);

    if ((conn_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        exit(3);

    server_addr.sin_family=AF_INET;
    server_addr.sin_port=htons(port);
    server_addr.sin_addr.s_addr=inet_addr(addr);

    if (connect(conn_sock, (struct sockaddr *)&server_addr, sizeof (server_addr)) < 0)
        exit(3);

    char buff[4096];
    char* cad = strdup("hola\r\n");
    char* quit = strdup("quit\r\n");
    uint64_t off, nanoseconds, start, time;
    long long int final;

    start = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

    for (int i = 0; i < num; i++)
    {
        nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();

        if (send(conn_sock, cad, strlen(cad), 0) > 0)
        {
            memset(buff, 0, sizeof(buff));

            if (recv(conn_sock, buff, sizeof(buff), 0) > 0)
            {
                off = strtoull(buff, NULL, 0);
                time = nanoseconds - start;
                final = off - nanoseconds;
                printf("%llu;%llu;%llu;%lld;%s\n", time, nanoseconds, off, final, addr);
            }
        }
        sleep(1);
    }
    send(conn_sock, quit, strlen(quit), 0);
    return 0;
}
