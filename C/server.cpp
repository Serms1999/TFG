#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <chrono>

#define BUFSIZE             8096
#define TIMEOUT_SEC         30L
#define TIMEOUT_USEC        0L

char* multi_tok(char* input, char** string, char* delimiter)
{
	if (input != NULL)
		*string = input;

	if (*string == NULL)
		return *string;

	char* end = strstr(*string, delimiter);
	if (end == NULL) {
		char* temp = *string;
		*string = NULL;
		return temp;
	}

	char* temp = *string;

	*end = '\0';
	*string = end + strlen(delimiter);
	return temp;
}

void process_web_request(int descriptorFichero)
{
    char buffer[BUFSIZE];
    char* line;
    char* rest;
    struct timeval tv;
    fd_set rfds;
    int retval;

    FD_ZERO(&rfds);
    FD_SET(descriptorFichero, &rfds);

    tv.tv_sec = TIMEOUT_SEC;
    tv.tv_usec = TIMEOUT_USEC;
    
    while ((retval = select(descriptorFichero + 1, &rfds, NULL, NULL, &tv)) != 0)
    {
        // Si select falla, el hilo termina
        if (retval == -1)
        {
            close(descriptorFichero);
            exit(1);
        }

        if (read(descriptorFichero, buffer, BUFSIZE) == -1)
        {
            close(descriptorFichero);
            exit(1);
        }

        char* sep = strdup("\r\n");
        line = multi_tok(buffer, &rest, sep);

        if (!strcmp(line, "hola"))
        {
            memset(buffer, 0, BUFSIZE);

            uint64_t nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::steady_clock::now().time_since_epoch()).count();

            sprintf(buffer, "%llu", nanoseconds);
            if ((write(descriptorFichero, buffer, strlen(buffer))) == -1) 
            {
                close(descriptorFichero);
                exit(1);
            }
        }
        else if (!strcmp(line, "quit") || !strcmp(line, "q"))
        {
            close(descriptorFichero);
            exit(0);
        }
        tv.tv_sec = TIMEOUT_SEC;
        tv.tv_usec = TIMEOUT_USEC;
    }
}

int main(int argc, char** argv)
{
    int port, pid, listenfd, socketfd;
    socklen_t length;
    static struct sockaddr_in cli_addr;        // static = Inicializado con ceros
    static struct sockaddr_in serv_addr;    // static = Inicializado con ceros

    // Hacemos que el proceso sea un demonio sin hijos zombies
    if (fork() != 0)
        return 0; // El proceso padre devuelve un OK al shell

    (void)signal(SIGCHLD, SIG_IGN); // Ignoramos a los hijos
    (void)signal(SIGHUP, SIG_IGN); // Ignoramos cuelgues

    /* setup the network socket */
    if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        exit(3);

    port = atoi(argv[1]);

    if (port < 0 || port >60000)
        exit(3);

    /*Se crea una estructura para la información IP y puerto donde escucha el servidor*/
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY); /*Escucha en cualquier IP disponible*/
    serv_addr.sin_port = htons(port); /*... en el puerto port especificado como parámetro*/

    if (bind(listenfd, (struct sockaddr*) & serv_addr, sizeof(serv_addr)) < 0)
        exit(3);

    if (listen(listenfd, 64) < 0)
        exit(3);

    while (1)
    {
        length = sizeof(cli_addr);
        if ((socketfd = accept(listenfd, (struct sockaddr*) & cli_addr, &length)) < 0)
            exit(3);
        if ((pid = fork()) < 0) {
            exit(3);
        }
        else
        {
            if (pid == 0)
            {     // Proceso hijo
                (void)close(listenfd);
                process_web_request(socketfd); // El hijo termina tras llamar a esta función
            }
            else
            {     // Proceso padre
                (void)close(socketfd);
            }
        }
    }
}
