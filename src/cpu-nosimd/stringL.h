// //packing dna strings
// //dna 5 has the N, dna has only acgt
// #include<string>

// template <typename TSpec = void>
// struct Alloc {};

// template <typename TValue, typename TSpec = Alloc<> >
// class loganString;

// template <typename TValue>
// struct BitsPerValueL
// {
//     static const unsigned VALUE = sizeof(TValue) * 8;
//     typedef unsigned Type;
// };

// template <typename TValue>
// struct BitsPerValueL<TValue const> : public BitsPerValueL<TValue>
// {};

// template <typename TValue>
// struct BytesPerValueL
// {
//     enum { VALUE = (BitsPerValueL<TValue>::VALUE + 7) / 8 };
// };

// template <typename T>
// struct ValueSizeL
// {
//     typedef uint64_t  Type;
//     static const Type VALUE = (BitsPerValueL<T>::VALUE < 64)? 1ull << (BitsPerValueL<T>::VALUE & 63) : 0ull;
// };

// template <typename T>
// struct ValueSizeL<T const> : ValueSizeL<T>
// {};

// #pragma pack(push,1)
// template <typename TValue, typename TSpec>
// class SimpleTypeL
// {
// public:
//     // ------------------------------------------------------------------------
//     // Members;  Have to be defined in class.
//     // ------------------------------------------------------------------------

//     TValue value;

//     // ------------------------------------------------------------------------
//     // Constructors;  Have to be defined in class.
//     // ------------------------------------------------------------------------

//     // TODO(holtgrew): Do we want default initialization?
   
//     SimpleTypeL() : value(0)
//     {}

   
//     SimpleTypeL(SimpleTypeL const & other)
//     {
//         assign(*this, other);
//     }

//     // TODO(holtgrew): Do we want an explicit here?
//     template <typename T>
   
//     SimpleTypeL(T const & other)
//     {
//         assign(*this, other);
//     }

//     // ------------------------------------------------------------------------
//     // Assignment Operator;  Have to be defined in class.
//     // ------------------------------------------------------------------------

   
//     SimpleTypeL & operator=(SimpleTypeL const & other)
//     {
//         assign(*this, other);
//         return *this;
//     }

//     template <typename T>
//     inline SimpleTypeL &
//     operator=(T const & other)
//     {
//         assign(*this, other);
//         return *this;
//     }

//     // ------------------------------------------------------------------------
//     // Conversion Operators;  Have to be defined in class.
//     // ------------------------------------------------------------------------

//     // Class.SimpleType specifies type conversion operators for all built-in
//     // integer types since there is no way to extend the build-in types with
//     // copy and assignment constructors in C++.
//     //
//     // This cannot be a template since it would conflict to the template
//     // constructor.

   
//     operator int64_t() const
//     {
//         int64_t c;
//         assign(c, *this);
//         return c;
//     }

   
//     operator uint64_t() const
//     {
//         uint64_t c;
//         assign(c, *this);
//         return c;
//     }

   
//     operator int() const
//     {
//         int c;
//         assign(c, *this);
//         return c;
//     }

   
//     operator unsigned int() const
//     {
//         unsigned int c;
//         assign(c, *this);
//         return c;
//     }

   
//     operator short() const
//     {
//         short c;
//         assign(c, *this);
//         return c;
//     }

   
//     operator unsigned short() const
//     {
//         unsigned short c;
//         assign(c, *this);
//         return c;
//     }

   
//     operator char() const
//     {
//         char c;
//         assign(c, *this);
//         return c;
//     }

   
//     operator signed char() const
//     {
//         signed char c;
//         assign(c, *this);
//         return c;
//     }


//     operator unsigned char() const
//     {
//         unsigned char c;
//         assign(c, *this);
//         return c;
//     }
// };
// #pragma pack(pop)


// template <typename THost>
// struct PrefixL
// {
//     typedef Segment<THost, PrefixSegment> Type;
// };

// template <typename THost>
// struct PrefixL< Segment<THost, InfixSegment> >
// {
//     typedef Segment<THost, InfixSegment> Type;
// };
// template <typename THost>
// struct PrefixL< Segment<THost, SuffixSegment> >
// {
//     typedef Segment<THost, InfixSegment> Type;
// };
// template <typename THost>
// struct PrefixL< Segment<THost, PrefixSegment> >
// {
//     typedef Segment<THost, PrefixSegment> Type;
// };

// template <typename THost, typename TSpec>
// struct PrefixL< Segment<THost, TSpec> const >:
//     PrefixL< Segment<THost, TSpec> > {};

// template <typename THost>
// struct PrefixL<THost &>:
//     PrefixL<THost> {};

// template <typename T, typename TPosEnd>
// inline typename PrefixL<T>::Type
// prefix(T & t, TPosEnd pos_end)
// {
//     return typename PrefixL<T>::Type(t, pos_end);
// }

// template <typename T, typename TPosEnd>
// inline typename PrefixL<T const>::Type
// prefix(T const & t, TPosEnd pos_end)
// {
//     return typename PrefixL<T const>::Type(t, pos_end);
// }

// template <typename T, typename TPosEnd>
// inline typename PrefixL<T *>::Type
// prefix(T * t, TPosEnd pos_end)
// {
//     return typename PrefixL<T *>::Type (t, pos_end);
// }

// struct DnaL_ {};
// typedef SimpleTypeL<unsigned char, DnaL_> DnaL;

// template <>
// struct ValueSizeL<DnaL>
// {
//     typedef uint8_t Type;
//     static const Type VALUE = 4;
// };

// template <>
// struct BitsPerValueL<DnaL>
// {
//     typedef uint8_t Type;
//     static const Type VALUE = 2;
// };

// struct Dna5L_ {};
// typedef SimpleTypeL<unsigned char, Dna5L_> Dna5L;

// template <>
// struct ValueSizeL<Dna5L>
// {
//     typedef uint8_t Type;
//     static const Type VALUE = 5;
// };

// template <>
// struct BitsPerValueL<Dna5L>
// {
//     typedef uint8_t Type;
//     static const Type VALUE = 3;
// };

// inline Dna5L
// unknownValueImpl(Dna5L *)
// {
//     static const Dna5L _result = Dna5L('N');
//     return _result;
// }

// typedef loganString<Dna5L, Alloc<void> > Dna5StringL;

// typedef loganString<DnaL, Alloc<void> > DnaString;