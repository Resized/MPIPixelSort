// Rotem Levi	205785959
// Yarden Shai	309920767
#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MASTER 0
#define RGB_SIZE 3 
#define DIMS 2 
#define ROW 0
#define COL 1

const char PATH[100] = "C:\\Program Files\\MPICH2\\bin\\pixel.txt";

struct Pixel {
	int id;
	int x;
	int y;
	float rgb[3];
};

enum CartPassDirection {
	COLS = 0,
	ROWS = 1,
};

enum CommDirection {
	RECEIVING = 0,
	SENDING = 1,
	NO_COMM = -1
};

enum SortDirection {
	ASCENDING = 0,
	DESCENDING = 1
};

void			create_pixel_mpi_type(Pixel* pixel, MPI_Datatype* PixelMPIType);
Pixel*			read_pixels_from_file(const char* file_path, int* num_of_pixels);
void			print_pixels(Pixel* pixels, int num_of_pixels);
void			print_row_pixels(Pixel* pixels_line, int count, bool is_forward);
void			print_shear_result(Pixel* pixels, int num_of_pixels, int matrix_size);
void			print_single_pixel(Pixel* p);
void			shearsort(Pixel* receivedPixel, MPI_Comm comm, int size, MPI_Datatype PixelMPIType);
void			odd_even_sort(int* coord, Pixel* storedPixel, CartPassDirection direction, MPI_Comm comm, int size, MPI_Datatype PixelMPIType);
void			exchange_between_neighbors(Pixel* storedPixel, CommDirection commDirection, SortDirection sortDirection, int neighborRank, MPI_Comm comm, MPI_Datatype PixelMPIType);
bool			pixel_compare(Pixel* pix1, Pixel* pix2, SortDirection sortDirection);
bool			is_pixel_zero(Pixel* pixel);
CommDirection	get_comm_direction(int* coord, int iteration, CartPassDirection direction);
SortDirection	get_sort_direction(int* coord, CartPassDirection direction);


int main(int argc, char *argv[]) {

	const char* file_path = PATH; // path to the pixels file
	Pixel *pixels = NULL;  // array of pixels
	int num_of_pixels, numprocs, rank;
	int dims[DIMS], period[DIMS] = { 0,0 }, reorder = 0;

	MPI_Comm cart_comm;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// creating a new MPI type - Pixel 
	struct Pixel pixel, *result;
	MPI_Datatype PixelMPIType;
	create_pixel_mpi_type(&pixel, &PixelMPIType);

	// master reads from txt file and prints pixels
	if (rank == MASTER)
	{
		pixels = read_pixels_from_file(file_path, &num_of_pixels);
		printf("Pixels before sort:\n");
		print_pixels(pixels, num_of_pixels);
		result = (Pixel*)malloc(sizeof(Pixel)* num_of_pixels);
	}

	//master sends the number of pixels to all processes 
	MPI_Bcast(&num_of_pixels, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

	//master sends each process one pixel
	MPI_Scatter(pixels, 1, PixelMPIType, &pixel, 1, PixelMPIType, MASTER, MPI_COMM_WORLD);

	//create cartesian topology 
	double square = sqrt(num_of_pixels);
	dims[0] = dims[1] = (int)square;
	MPI_Cart_create(MPI_COMM_WORLD, DIMS, dims, period, reorder, &cart_comm);

	//sort by required order
	shearsort(&pixel, cart_comm, dims[0], PixelMPIType);

	//master receives data from each process
	MPI_Gather(&pixel, 1, PixelMPIType, result, 1, PixelMPIType, MASTER, cart_comm);

	if (rank == MASTER)
	{
		printf("\n\nPixels after sort:\n");
		print_shear_result(result, num_of_pixels, dims[0]);
		free(result);
		free(pixels);
	}

	MPI_Finalize();
	return 0;
}

// creating a new MPI type - Pixel 
void create_pixel_mpi_type(Pixel* pixel, MPI_Datatype* PixelMPIType)
{
	MPI_Datatype type[4] = { MPI_INT, MPI_INT, MPI_INT, MPI_FLOAT };
	int blocklen[4] = { 1, 1, 1 , RGB_SIZE };
	MPI_Aint disp[4];
	disp[0] = (char *)&pixel->id - (char *)pixel;
	disp[1] = (char *)&pixel->x - (char *)pixel;
	disp[2] = (char *)&pixel->y - (char *)pixel;
	disp[3] = (char *)&pixel->rgb - (char *)pixel;
	MPI_Type_create_struct(4, blocklen, disp, type, PixelMPIType);
	MPI_Type_commit(PixelMPIType);
}

// Reads pixels from a given file path and allocates and returns pixels array
Pixel *read_pixels_from_file(const char* file_path, int* num_of_pixels)
{
	FILE *file;
	Pixel  *pixels;
	Pixel current;
	int j;

	//open file to read
	file = fopen(file_path, "r");

	//failed to open file 
	if (file == NULL)
	{
		printf("file not opened - make sure the .txt file is in the correct path\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	//read first line - number of pixels
	fscanf(file, "%d", num_of_pixels);

	//allocate an array of pixels with the size num_of_pixels
	pixels = (Pixel*)malloc(sizeof(Pixel)*(*num_of_pixels));
	Pixel *ptr;
	// read the pixels to the array 
	for (ptr = pixels; ptr < pixels + *num_of_pixels; ptr++)
	{
		fscanf(file, "%d", &current.id);
		fscanf(file, "%d", &current.x);
		fscanf(file, "%d", &current.y);
		for (j = 0; j < RGB_SIZE; j++)
			fscanf(file, "%f", &current.rgb[j]);
		*ptr = current;
	}

	return pixels;
}

void print_pixels(Pixel* pixels, int num_of_pixels)
{
	Pixel* ptr;
	int i;
	char rgb[4] = "RGB";

	printf("%-3s%5s%5s", "ID", "X", "Y");
	for (i = 0; i < 3; i++)
		printf("%6c", rgb[i]);
	printf("\n");

	for (ptr = pixels; ptr < pixels + num_of_pixels; ptr++)
	{
		print_single_pixel(ptr);
	}

}

void print_shear_result(Pixel* pixels, int num_of_pixels, int matrix_size)
{
	bool even_lines_swapper = true;
	int i;
	char rgb[4] = "RGB";

	printf("%-3s%5s%5s", "ID", "X", "Y");
	for (i = 0; i < 3; i++)
		printf("%6c", rgb[i]);
	printf("\n");

	for (int i = 0; i < num_of_pixels; i += matrix_size)
	{
		print_row_pixels(pixels + i, matrix_size, even_lines_swapper);
		even_lines_swapper = !even_lines_swapper;
	}
}

void print_row_pixels(Pixel* pixels_line, int count, bool is_forward)
{
	Pixel* ptr;

	if (is_forward)
		for (ptr = pixels_line; ptr < pixels_line + count; ptr++)
			print_single_pixel(ptr);
	else
		for (ptr = pixels_line + count - 1; ptr >= pixels_line; ptr--)
			print_single_pixel(ptr);
}

void print_single_pixel(Pixel* p) {
	int i;
	printf("%-3d", p->id);
	printf("%5d", p->x);
	printf("%5d", p->y);
	for (i = 0; i < RGB_SIZE; i++)
		printf("%6.2f", p->rgb[i]);
	printf("\n");
}

void shearsort(Pixel* receivedPixel, MPI_Comm comm, int size, MPI_Datatype PixelMPIType)
{
	int rank;
	int coord[2];
	MPI_Comm_rank(comm, &rank);
	MPI_Cart_coords(comm, rank, DIMS, coord);

	int totalIterations = (int)ceil(log2((double)size)) + 1;

	for (int i = 0; i <= totalIterations; i++)
	{
		// Rows pass
		odd_even_sort(coord, receivedPixel, ROWS, comm, size, PixelMPIType);
		// Columns pass
		odd_even_sort(coord, receivedPixel, COLS, comm, size, PixelMPIType);
	}
}

void odd_even_sort(int* coord, Pixel* storedPixel, CartPassDirection passDirection, MPI_Comm comm, int size, MPI_Datatype PixelMPIType)
{
	int neighbor1, neighbor2, neighborRankForExchange;
	CommDirection commDirection;
	SortDirection sortDirection = get_sort_direction(coord, passDirection);
	MPI_Cart_shift(comm, passDirection, 1, &neighbor1, &neighbor2);

	for (int i = 0; i < size; i++)
	{
		commDirection = get_comm_direction(coord, i, passDirection);
		neighborRankForExchange = commDirection == SENDING ? neighbor2 : neighbor1;

		if (neighborRankForExchange != MPI_PROC_NULL) // Exchange only if we in bounds
			exchange_between_neighbors(storedPixel, commDirection, sortDirection, neighborRankForExchange, comm, PixelMPIType);
	}
}

void exchange_between_neighbors(Pixel* storedPixel, CommDirection commDirection, SortDirection sortDirection, int neighborRank, MPI_Comm comm, MPI_Datatype PixelMPIType)
{
	MPI_Status status;
	Pixel temp;
	if (commDirection == SENDING) // Sending side
	{
		MPI_Send(storedPixel, 1, PixelMPIType, neighborRank, 0, comm);
		MPI_Recv(storedPixel, 1, PixelMPIType, neighborRank, 0, comm, &status);
	}
	else // Receiving side. Make the actual check/sort
	{
		Pixel received;
		MPI_Recv(&received, 1, PixelMPIType, neighborRank, 0, comm, &status);
		if (pixel_compare(storedPixel, &received, sortDirection))
		{
			temp = *storedPixel;
			*storedPixel = received;
			received = temp;
		}
		MPI_Send(&received, 1, PixelMPIType, neighborRank, 0, comm);
	}
}

bool pixel_compare(Pixel* pix1, Pixel* pix2, SortDirection sortDirection)
{
	Pixel temp1 = *pix1, temp2 = *pix2;
	float zeroes[3] = { 0,0,0 };

	if (is_pixel_zero(&temp1) && is_pixel_zero(&temp2))
	{
		double d1 = sqrt(temp1.x*temp1.x + temp1.y*temp1.y);
		double d2 = sqrt(temp2.x*temp2.x + temp2.y*temp2.y);
		double d_diff = d1 - d2;
		return sortDirection == ASCENDING ? d_diff > 0 : d_diff < 0;
	}
	float sum_diff = 0, sum1 = 0, sum2 = 0;
	int i;
	for (i = 0; i < RGB_SIZE; i++)
	{
		sum1 += temp1.rgb[i];
		sum2 += temp2.rgb[i];
	}
	sum_diff = sum1 - sum2;
	return sortDirection == ASCENDING ? sum_diff > 0 : sum_diff < 0;
}

bool is_pixel_zero(Pixel* pixel)
{
	int i;
	Pixel temp = *pixel;
	for (i = 0; i < RGB_SIZE; i++)
	{
		if (!temp.rgb[i] == 0)
			return false;
	}
	return true;
}

CommDirection get_comm_direction(int* coord, int iteration, CartPassDirection direction)
{
	// if position even and iteration even we are sending, otherwise receiving
	// if position odd and iteration odd we are sending, otherwise receiving
	return (iteration % 2 == coord[direction] % 2) ? SENDING : RECEIVING;
}

SortDirection get_sort_direction(int* coord, CartPassDirection direction)
{
	// Even Row	  ASCENDING
	// Odd Row    DESCENDING
	// Col always ASCENDING;
	if (direction == COLS) return ASCENDING;
	return (SortDirection)(coord[0] % 2);
}
